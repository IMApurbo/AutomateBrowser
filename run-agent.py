import logging
import os
import asyncio
import argparse
import traceback
import requests

from dotenv import load_dotenv
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextWindowSize, BrowserContextConfig
from src.utils.agent_state import AgentState
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.controller.custom_controller import CustomController

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global variables
_global_browser = None
_global_browser_context = None
_global_agent = None
_global_agent_state = AgentState()

# File to save API key
API_KEY_FILE = "api_key.txt"

# Resolve sensitive environment variables
def resolve_sensitive_env_variables(text):
    if not text:
        return text
    import re
    env_vars = re.findall(r'\$SENSITIVE_[A-Za-z0-9_]*', text)
    result = text
    for var in env_vars:
        env_name = var[1:]
        env_value = os.getenv(env_name)
        if env_value is not None:
            result = result.replace(var, env_value)
    return result

# Load saved API key
def load_api_key(provider):
    file_path = f"{provider}_api_key.txt"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return f.read().strip()
    return ""

# Save API key
def save_api_key(api_key, provider):
    file_path = f"{provider}_api_key.txt"
    with open(file_path, "w") as f:
        f.write(api_key)
    return f"API key for {provider} saved successfully!"

# Check installed Ollama models
def get_ollama_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        else:
            return ["No models found"]
    except Exception as e:
        logger.error(f"Error checking Ollama models: {e}")
        return ["Ollama not running or no models installed"]

# Get LLM model for Groq, Google, and Ollama
def get_llm_model(provider: str, **kwargs):
    if provider == "groq":
        api_key = kwargs.get("api_key", "") or os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise gr.Error("💥 Groq API key not found! 🔑 Please set the `GROQ_API_KEY` environment variable or provide it in the UI.")
        base_url = kwargs.get("base_url", "https://api.groq.com/openai/v1")
        return ChatOpenAI(
            model=kwargs.get("model_name", "llama-3.3-70b-versatile"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
            max_tokens=kwargs.get("num_ctx", 8192)
        )
    elif provider == "google":
        api_key = kwargs.get("api_key", "") or os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise gr.Error("💥 Google API key not found! 🔑 Please set the `GOOGLE_API_KEY` environment variable or provide it in the UI.")
        base_url = kwargs.get("base_url", "https://generativelanguage.googleapis.com")
        return ChatGoogleGenerativeAI(
            model=kwargs.get("model_name", "gemini-2.0-flash"),
            temperature=kwargs.get("temperature", 0.0),
            google_api_key=api_key,
            base_url=base_url,
            max_output_tokens=kwargs.get("num_ctx", 8192)
        )
    elif provider == "ollama":
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return ChatOllama(
            model=kwargs.get("model_name", "qwen2.5:7b"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            num_ctx=kwargs.get("num_ctx", 32000),
            num_predict=kwargs.get("num_predict", 1024)
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# Predefined models
model_names = {
    "groq": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma-7b-it"],
    "google": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    "ollama": get_ollama_models()
}

# Update model dropdown based on provider
def update_model_dropdown(llm_provider):
    if llm_provider in model_names:
        choices = model_names[llm_provider]
        value = choices[0] if choices else ""
        return gr.Dropdown(choices=choices, value=value, interactive=True, allow_custom_value=(llm_provider == "ollama"))
    return gr.Dropdown(choices=[], value="", interactive=True, allow_custom_value=True)

# Highlight element in the browser
async def highlight_element(page, selector):
    try:
        # Inject CSS to highlight the element
        await page.evaluate("""
            (selector) => {
                const element = document.querySelector(selector);
                if (element) {
                    element.style.outline = '3px solid red';
                    element.style.backgroundColor = 'rgba(255, 0, 0, 0.1)';
                    setTimeout(() => {
                        element.style.outline = '';
                        element.style.backgroundColor = '';
                    }, 2000);  // Remove highlight after 2 seconds
                }
            }
        """, selector)
    except Exception as e:
        logger.error(f"Error highlighting element: {e}")

# Wrapper around CustomAgent to add highlighting
class HighlightingCustomAgent(CustomAgent):
    async def click(self, selector, **kwargs):
        page = self.browser_context.pages[0]  # Assume single page
        await highlight_element(page, selector)
        await super().click(selector, **kwargs)

    async def fill(self, selector, value, **kwargs):
        page = self.browser_context.pages[0]
        await highlight_element(page, selector)
        await super().fill(selector, value, **kwargs)

# Stop agent
async def stop_agent():
    global _global_agent, _global_agent_state, _global_browser, _global_browser_context
    try:
        if _global_agent:
            _global_agent.stop()
        _global_agent_state.request_stop()
        message = "Agent stopped successfully!"
        logger.info(f"🛑 {message}")
        
        if _global_browser_context:
            await _global_browser_context.close()
            _global_browser_context = None
        if _global_browser:
            await _global_browser.close()
            _global_browser = None
        _global_agent = None
        
        return message, gr.update(interactive=True), gr.update(interactive=True)
    except Exception as e:
        error_msg = f"Error stopping agent: {str(e)}"
        logger.error(error_msg)
        return error_msg, gr.update(interactive=True), gr.update(interactive=True)

# Run browser agent with highlighting
async def run_browser_agent(
    provider, api_key, model_name, temperature, num_ctx,
    window_w, window_h, task, max_steps, use_vision, max_actions_per_step
):
    global _global_browser, _global_browser_context, _global_agent, _global_agent_state
    _global_agent_state.clear_stop()

    try:
        task = resolve_sensitive_env_variables(task)

        llm = get_llm_model(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            num_ctx=num_ctx
        )

        _global_browser = CustomBrowser(
            config=BrowserConfig(
                headless=False,
                disable_security=False,
                extra_chromium_args=[f"--window-size={window_w},{window_h}"]
            )
        )

        _global_browser_context = await _global_browser.new_context(
            config=BrowserContextConfig(
                no_viewport=False,
                browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h)
            )
        )

        controller = CustomController()
        _global_agent = HighlightingCustomAgent(  # Use the highlighting wrapper
            task=task,
            add_infos="",
            use_vision=use_vision,
            llm=llm,
            browser=_global_browser,
            browser_context=_global_browser_context,
            controller=controller,
            system_prompt_class=CustomSystemPrompt,
            agent_prompt_class=CustomAgentMessagePrompt,
            max_actions_per_step=max_actions_per_step
        )

        history = await _global_agent.run(max_steps=max_steps)
        if _global_agent_state.is_stop_requested():
            raise Exception("Agent was stopped by user.")

        history_file = os.path.join("./tmp/agent_history", f"{_global_agent.agent_id}.json")
        os.makedirs("./tmp/agent_history", exist_ok=True)
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        return (
            final_result, errors, model_actions, model_thoughts, history_file,
            gr.update(interactive=True), gr.update(interactive=True)
        )

    except Exception as e:
        errors = f"Error: {str(e)}\n{traceback.format_exc()}"
        logger.error(errors)
        return (
            "", errors, "", "", None,
            gr.update(interactive=True), gr.update(interactive=True)
        )
    finally:
        if _global_browser_context:
            await _global_browser_context.close()
            _global_browser_context = None
        if _global_browser:
            await _global_browser.close()
            _global_browser = None
        _global_agent = None

# Wrapper for Gradio
def run_agent_wrapper(
    provider, api_key, model_name, temperature, num_ctx,
    window_w, window_h, task, max_steps, use_vision, max_actions_per_step
):
    result = asyncio.run(run_browser_agent(
        provider, api_key, model_name, temperature, num_ctx,
        window_w, window_h, task, max_steps, use_vision, max_actions_per_step
    ))
    return result

# Create Gradio UI
def create_ui():
    css = """
    .gradio-container { max-width: 1200px !important; margin: auto !important; padding-top: 20px !important; }
    .header-text { text-align: center; margin-bottom: 30px; }
    """
    
    with gr.Blocks(title="Browser Agent", css=css) as demo:
        gr.Markdown("## 🌐 Browser Agent with Highlighting", elem_classes=["header-text"])

        with gr.Tabs():
            # LLM Settings Tab
            with gr.TabItem("🔧 LLM Settings"):
                provider = gr.Dropdown(
                    choices=["groq", "google", "ollama"],
                    label="LLM Provider",
                    value="groq"
                )
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="e.g., gsk_xxxx (Groq) or AIzaSy... (Google)",
                    value=load_api_key("groq")
                )
                save_api_key_button = gr.Button("💾 Save API Key")
                api_key_status = gr.Textbox(label="API Key Status", interactive=False)
                model_name = gr.Dropdown(
                    label="Model Name",
                    choices=model_names["groq"],
                    value="llama-3.3-70b-versatile"
                )
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature")
                num_ctx = gr.Slider(256, 131072, value=8192, step=1, label="Max Context Length")

            # Browser Settings Tab
            with gr.TabItem("🌐 Browser Settings"):
                window_w = gr.Number(label="Window Width", value=1280)
                window_h = gr.Number(label="Window Height", value=720)

            # Run Agent Tab
            with gr.TabItem("🤖 Run Agent"):
                task = gr.Textbox(label="Task", lines=4, placeholder="e.g., Search for AI news")
                max_steps = gr.Slider(1, 100, value=10, step=1, label="Max Steps")
                use_vision = gr.Checkbox(label="Use Vision", value=False)
                max_actions_per_step = gr.Slider(1, 20, value=5, step=1, label="Max Actions per Step")
                
                with gr.Row():
                    run_button = gr.Button("▶️ Run Agent", variant="primary")
                    stop_button = gr.Button("⏹️ Stop", variant="stop")

            # Results Tab
            with gr.TabItem("📊 Results"):
                final_result_output = gr.Textbox(label="Final Result", lines=3)
                errors_output = gr.Textbox(label="Errors", lines=3)
                model_actions_output = gr.Textbox(label="Model Actions", lines=3)
                model_thoughts_output = gr.Textbox(label="Model Thoughts", lines=3)
                history_file = gr.File(label="Agent History")

                save_api_key_button.click(
                    fn=save_api_key,
                    inputs=[api_key, provider],
                    outputs=[api_key_status]
                )
                provider.change(
                    fn=update_model_dropdown,
                    inputs=[provider],
                    outputs=[model_name]
                )
                provider.change(
                    fn=lambda p: gr.update(value=load_api_key(p)),
                    inputs=[provider],
                    outputs=[api_key]
                )
                run_button.click(
                    fn=run_agent_wrapper,
                    inputs=[
                        provider, api_key, model_name, temperature, num_ctx,
                        window_w, window_h, task, max_steps, use_vision, max_actions_per_step
                    ],
                    outputs=[
                        final_result_output, errors_output, model_actions_output,
                        model_thoughts_output, history_file, run_button, stop_button
                    ]
                )
                stop_button.click(
                    fn=stop_agent,
                    inputs=[],
                    outputs=[errors_output, run_button, stop_button]
                )

    return demo

def main():
    parser = argparse.ArgumentParser(description="Browser Agent UI")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address")
    parser.add_argument("--port", type=int, default=7788, help="Port")
    args = parser.parse_args()

    demo = create_ui()
    demo.launch(server_name=args.ip, server_port=args.port)

if __name__ == "__main__":
    main()

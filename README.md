# 🤖 Browser Agent – AI-Powered Web Automation with Visual Highlighting

A powerful browser automation agent that uses LLMs (Groq, Google Gemini, or Ollama) to understand and execute web tasks. Features **real-time visual highlighting** of clicked/filled elements, multi-provider support, and a clean Gradio UI.

> **⚠️ Disclaimer**  
> This tool automates browser interactions. Use responsibly and only on websites you own or have permission to test. The author (`IMApurbo`) is not liable for misuse or violations of website terms of service.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![LangChain](https://img.shields.io/badge/LangChain-0.3+-green) ![Gradio](https://img.shields.io/badge/Gradio-5.0+-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multi-LLM Support** | Groq, Google Gemini, and Ollama (local) |
| **Visual Highlighting** | Clicked/filled elements flash red for visibility |
| **Browser Automation** | Full control via `browser-use` framework |
| **Custom Prompts** | Extensible system and agent prompts |
| **Task Execution** | Run natural language tasks (e.g., "Search for AI news") |
| **Agent State Management** | Stop/resume capabilities |
| **History Export** | Save execution history as JSON |
| **Sensitive Variable Support** | Use `$SENSITIVE_VAR` in tasks for secrets |
| **API Key Management** | Save/load keys per provider |

---

## 🎯 Demo

```yaml
Task: "Go to github.com, search for 'browser-use', and click the first result"
Agent: Opens browser → Navigates → Types in search → Clicks result (highlighted in red)
Output: Success message + detailed action log
```

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- Playwright browsers
- Optional: Ollama (for local models)

### Quick Install

```bash
git clone https://github.com/IMApurbo/browser-agent.git
cd browser-agent
pip install -r requirements.txt
playwright install  # Install browser binaries
```

### Required Dependencies

```txt
langchain-openai
langchain-google-genai
langchain-ollama
browser-use
gradio
python-dotenv
requests
beautifulsoup4
playwright
```

---

## 🚀 Usage

### 1. Set Environment Variables (Optional)

Create a `.env` file:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxx
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxx
SENSITIVE_PASSWORD=mySecret123
```

> Use `$SENSITIVE_PASSWORD` in tasks to reference sensitive values.

### 2. Run the Application

```bash
python run-agent.py --ip 127.0.0.1 --port 7788
```

### 3. Open the UI

Navigate to `http://127.0.0.1:7788` in your browser.

---

## 🖥️ UI Walkthrough

### Tab 1: LLM Settings

| Field | Description |
|-------|-------------|
| **LLM Provider** | Choose Groq, Google, or Ollama |
| **API Key** | Enter your key (saved per provider) |
| **Model Name** | Select from available models |
| **Temperature** | 0.0 (deterministic) to 2.0 (creative) |
| **Max Context Length** | Token limit for the model |

### Tab 2: Browser Settings

| Field | Description |
|-------|-------------|
| **Window Width** | Browser window width (default: 1280) |
| **Window Height** | Browser window height (default: 720) |

### Tab 3: Run Agent

| Field | Description |
|-------|-------------|
| **Task** | Natural language instruction |
| **Max Steps** | Maximum execution steps (1-100) |
| **Use Vision** | Enable visual understanding (slower) |
| **Max Actions per Step** | Actions before re-prompting |

### Tab 4: Results

- **Final Result** – Outcome of the task
- **Errors** – Any exceptions encountered
- **Model Actions** – JSON of performed actions
- **Model Thoughts** – LLM reasoning log
- **Agent History** – Downloadable JSON file

---

## 🔧 Example Tasks

### Basic Navigation
```
Go to https://example.com and take a screenshot
```

### Form Filling
```
Navigate to https://github.com/login, fill username 'testuser', fill password $SENSITIVE_PASSWORD, click sign in
```

### Data Extraction
```
Open https://news.ycombinator.com, extract the titles of the top 5 posts, save to a text file
```

### E-commerce Automation
```
Search Amazon for 'wireless mouse', sort by price low to high, get the first result's price
```

---

## 🎨 Highlighting Feature

When the agent clicks or fills an element:

```javascript
// Injected CSS
element.style.outline = '3px solid red';
element.style.backgroundColor = 'rgba(255, 0, 0, 0.1)';

// Removed after 2 seconds
setTimeout(() => {
    element.style.outline = '';
    element.style.backgroundColor = '';
}, 2000);
```

This provides **real-time visual feedback** of what the agent is interacting with.

---

## 🧠 Supported LLM Providers

### Groq
| Model | Context | Speed |
|-------|---------|-------|
| `llama-3.3-70b-versatile` | 8,192 | Fastest |
| `mixtral-8x7b-32768` | 32,768 | Good |
| `gemma-7b-it` | 8,192 | Lightweight |

### Google Gemini
| Model | Context | Use Case |
|-------|---------|----------|
| `gemini-2.0-flash` | 8,192 | Fast, balanced |
| `gemini-1.5-flash` | 8,192 | Quick responses |
| `gemini-1.5-pro` | 8,192 | Complex reasoning |

### Ollama (Local)
| Model | Context | Requirements |
|-------|---------|--------------|
| `qwen2.5:7b` | 32,000 | 4GB RAM |
| `llama3.2:3b` | 8,192 | 2GB RAM |
| `phi3:mini` | 8,192 | 2GB RAM |

> Run Ollama: `ollama serve` then `ollama pull qwen2.5:7b`

---

## 📁 File Structure

```
browser-agent/
├── run-agent.py              # Main application
├── src/
│   ├── agent/
│   │   ├── custom_agent.py   # Extended agent class
│   │   └── custom_prompts.py # Custom prompt templates
│   ├── browser/
│   │   └── custom_browser.py # Extended browser class
│   ├── controller/
│   │   └── custom_controller.py # Action handlers
│   └── utils/
│       └── agent_state.py    # Stop/state management
├── tmp/
│   └── agent_history/        # Saved execution logs
├── groq_api_key.txt          # Saved API keys (auto-generated)
├── google_api_key.txt
└── .env                      # Environment variables
```

---

## 🛠️ Advanced Configuration

### Custom System Prompt

Modify `src/agent/custom_prompts.py`:

```python
class CustomSystemPrompt(SystemPrompt):
    def get_prompt(self) -> str:
        return """You are a helpful assistant that...
        (custom instructions here)
        """
```

### Sensitive Variable Resolution

Use in tasks:
```
Login with username 'admin' and password $SENSITIVE_PASSWORD
```

The variable will be replaced at runtime.

### Headless Mode

Modify `run-agent.py`:

```python
_global_browser = CustomBrowser(
    config=BrowserConfig(
        headless=True,  # Change to True
        disable_security=False,
    )
)
```

---

## 🐛 Troubleshooting

### ❌ "Groq API key not found"

- Set `GROQ_API_KEY` in `.env`
- Or enter key in UI and click "Save API Key"

### ❌ Ollama connection refused

```bash
# Start Ollama service
ollama serve

# Verify models are installed
ollama list

# Pull a model if needed
ollama pull qwen2.5:7b
```

### ❌ Playwright browser missing

```bash
playwright install chromium
```

### ❌ Agent gets stuck

- Click the **Stop** button in UI
- Reduce `max_steps` or `max_actions_per_step`
- Lower temperature for more deterministic behavior

---

## 🔒 Security Notes

- API keys are saved in **plaintext** files (`*_api_key.txt`)
- Use environment variables for production (`GROQ_API_KEY`, `GOOGLE_API_KEY`)
- The agent can execute arbitrary browser actions – **validate tasks before running**
- Sensitive variables are resolved in memory only

---

## 🧪 Testing

Run with a simple task:

```bash
python run-agent.py --ip 0.0.0.0 --port 7788
```

Then in UI:
- Provider: `groq`
- API Key: `your-key`
- Task: `Go to https://example.com and tell me the page title`
- Max Steps: `5`

Watch the browser open and execute the task with visual highlighting!

---

## 🤝 Contributing

Pull requests welcome for:

- Additional LLM providers (Anthropic, OpenAI, Cohere)
- Screenshot capture per step
- Recorder mode (record and replay actions)
- Parallel agent execution
- Docker deployment

---

## 📜 License

MIT License – see [LICENSE](LICENSE) file.

Copyright (c) 2025 **IMApurbo**

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

---

## 📬 Contact

**Author:** IMApurbo  
**GitHub:** [https://github.com/IMApurbo](https://github.com/IMApurbo)

---

> *Automate responsibly. Test ethically. Build amazing things.*
```


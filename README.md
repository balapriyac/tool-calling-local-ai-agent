# Local Tool-Calling Agent

A simple AI agent in plain Python that answers questions about a folder of your own files. It uses [Ollama](https://ollama.com) to run the model locally, so the files, the model, and the conversation stay on your machine. There is no agent framework and no API key.

## What it can do

The model chooses from five tools:

| Tool | Purpose |
| --- | --- |
| `list_files` | List the folders and text files in a folder |
| `search_files` | Find lines that contain a word or phrase |
| `read_file` | Read a numbered range of lines from a file |
| `recent_files` | List files changed in the last few days |
| `remember` | Save a fact for future sessions |

All paths are checked against the folder you point it at, so the model cannot read files outside it. Every tool result is size-limited, and errors come back as text the model can act on.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/download), installed and running
- A model that supports tool calling, such as `qwen3:4b`

## Setup

```bash
ollama pull qwen3:4b
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install ollama
```

## Example Usage

```bash
python assistant.py ~/my-project
```

Ask a question at the `You:` prompt, and type `exit` to quit. Each tool call prints as a `[tool]` line, so you can see what the model chose.

## Configuration

Settings are at the top of `assistant.py`:

- `MODEL`: the Ollama model to use
- `TEXT_SUFFIXES`: the file types the tools can read
- `MAX_RESULTS` and `MAX_TOOL_CHARS`: limits on the size of each tool result
- `MEMORY_FILE`: where remembered facts are stored (your home directory by default)

Agents fill the context window quickly, because every tool result joins the conversation. Ollama's default context length depends on your GPU memory and can be as low as 4,000 tokens, and the docs suggest a much larger 64,000 for agent workflows. Qwen3-4B has a native context of 32,768 tokens, so let's set a 32,000-token window to run our example. Set it in the Ollama app's settings, or start the server with:

```bash
OLLAMA_CONTEXT_LENGTH=32000 ollama serve
```

Run `ollama ps` to confirm the model is running on your GPU. If you hit the limit, lower `keep_turns` in `trim` to 4 and `MAX_TOOL_CHARS` to about 2,000.

## How it works

1. `run_turn` sends the conversation and the tool list to the model.
2. If the model asks for tools, the code runs each one and adds the results to the conversation.
3. The loop repeats until the model replies without a tool call, or stops after `max_steps`.
4. Short-term memory is the trimmed message list. Long-term memory is a facts file that the system prompt loads at startup.

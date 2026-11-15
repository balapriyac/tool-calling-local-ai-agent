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


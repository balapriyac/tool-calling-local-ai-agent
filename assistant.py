import json
import sys
import time
from datetime import datetime
from pathlib import Path

from ollama import chat

MODEL = "qwen3:4b"
ROOT = Path(".").resolve()
MEMORY_FILE = Path.home() / ".doc_assistant_memory.json"
TEXT_SUFFIXES = {".md", ".txt", ".csv", ".json", ".py", ".rst"}
MAX_RESULTS = 20
MAX_TOOL_CHARS = 3000


def safe_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("that path is outside the folder")
    return path


def text_files():
    for path in sorted(ROOT.rglob("*")):
        parts = path.relative_to(ROOT).parts
        if any(part.startswith(".") for part in parts):
            continue
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            if path.resolve().is_relative_to(ROOT):
                yield path


def list_files(subdirectory: str = ".") -> str:
    """List the folders and text files directly inside a folder, with file sizes.

    Args:
        subdirectory: The folder to list, relative to the root. Use "." for the top level.
    """
    try:
        folder = safe_path(subdirectory)
    except ValueError as error:
        return f"Error: {error}."
    if not folder.is_dir():
        return f"Error: '{subdirectory}' is not a folder."
    lines = []
    for path in sorted(folder.iterdir()):
        if path.name.startswith(".") or not path.resolve().is_relative_to(ROOT):
            continue
        if path.is_dir():
            lines.append(f"{path.name}/")
        elif path.suffix.lower() in TEXT_SUFFIXES:
            lines.append(f"{path.name}  ({path.stat().st_size} bytes)")
    return "\n".join(lines) or "The folder has no readable files."


def search_files(query: str, extension: str = "") -> str:
    """Search every text file for lines that contain a word or phrase, ignoring case.
    Each match is returned as path:line: text.

    Args:
        query: The word or phrase to look for
        extension: Optional file type filter such as ".md". Leave empty to search all text files.
    """
    needle = query.strip().lower()
    if not needle:
        return "Error: the query is empty."
    matches = []
    for path in text_files():
        if extension and path.suffix.lower() != extension.lower():
            continue
        text = path.read_text(errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if needle in line.lower():
                matches.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:160]}")
                if len(matches) == MAX_RESULTS:
                    return "\n".join(matches) + "\n[stopped at the match limit; narrow the query]"
    return "\n".join(matches) or f"No line contains '{query}'."


def read_file(path: str, start_line: int = 1, max_lines: int = 80) -> str:
    """Read part of a text file, with line numbers.

    Args:
        path: The file path relative to the root, as shown by list_files or search_files
        start_line: The first line to read, counting from 1
        max_lines: The most lines to return
    """
    try:
        file = safe_path(path)
    except ValueError as error:
        return f"Error: {error}."
    if not file.is_file() or file.suffix.lower() not in TEXT_SUFFIXES:
        return f"Error: '{path}' is not a readable text file."
    lines = file.read_text(errors="replace").splitlines()
    start = max(int(start_line), 1)
    chunk = lines[start - 1 : start - 1 + int(max_lines)]
    if not chunk:
        return f"No lines at that position. The file has {len(lines)} lines."
    numbered = "\n".join(f"{start + i}: {line}" for i, line in enumerate(chunk))
    return f"{numbered}\n[lines {start}-{start + len(chunk) - 1} of {len(lines)}]"


def recent_files(days: int = 7) -> str:
    """List the text files changed within the last number of days, newest first.

    Args:
        days: How many days back to look
    """
    cutoff = time.time() - int(days) * 86400
    found = [(p.stat().st_mtime, p) for p in text_files() if p.stat().st_mtime >= cutoff]
    found.sort(reverse=True)
    lines = [
        f"{datetime.fromtimestamp(stamp):%Y-%m-%d %H:%M}  {path.relative_to(ROOT)}"
        for stamp, path in found[:MAX_RESULTS]
    ]
    return "\n".join(lines) or f"No files changed in the last {days} days."


def load_memory() -> list:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    return []


def remember(fact: str) -> str:
    """Store a fact about the user or their work so it is available in future sessions.

    Args:
        fact: One short, self-contained statement, such as "The tests run with pytest -q"
    """
    facts = load_memory()
    if fact not in facts:
        facts.append(fact)
        MEMORY_FILE.write_text(json.dumps(facts, indent=2))
    return "Remembered."


TOOLS = {
    function.__name__: function
    for function in [list_files, search_files, read_file, recent_files, remember]
}


def clip(text: str) -> str:
    if len(text) <= MAX_TOOL_CHARS:
        return text
    return text[:MAX_TOOL_CHARS] + "\n[output cut; ask for a narrower range]"


def run_tool(name: str, arguments: dict) -> str:
    function = TOOLS.get(name)
    if function is None:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(TOOLS)}."
    try:
        return clip(str(function(**arguments)))
    except (TypeError, ValueError) as error:
        return f"Error: bad arguments ({error})."


def run_turn(messages: list, max_steps: int = 8) -> str:
    for _ in range(max_steps):
        response = chat(
            model=MODEL,
            messages=messages,
            tools=list(TOOLS.values()),
            options={"temperature": 0},
        )
        messages.append(response.message)

        if not response.message.tool_calls:
            return response.message.content

        for call in response.message.tool_calls:
            name = call.function.name
            arguments = call.function.arguments
            print(f"  [tool] {name}({arguments})")
            messages.append(
                {"role": "tool", "tool_name": name, "content": run_tool(name, arguments)}
            )

    return "Stopped: the agent used too many tool steps."


def role_of(message) -> str:
    return message["role"] if isinstance(message, dict) else message.role


def trim(messages: list, keep_turns: int = 6) -> list:
    user_positions = [i for i, m in enumerate(messages) if role_of(m) == "user"]
    if len(user_positions) <= keep_turns:
        return messages
    return [messages[0]] + messages[user_positions[-keep_turns]:]


def build_system_prompt() -> str:
    prompt = (
        "You are an assistant for a folder of the user's own project files. "
        "Search before you read, and read only the lines you need. "
        "Answer from what the files say, and cite each claim as path:line. "
        "If the files do not contain the answer, say so. "
        "You can call several tools in one turn, and you can keep calling tools "
        "until you have what you need. "
        "Call remember only when the user asks you to remember something."
    )
    facts = load_memory()
    if facts:
        prompt += "\n\nWhat you know about the user:\n" + "\n".join(f"- {f}" for f in facts)
    return prompt


def main():
    global ROOT
    if len(sys.argv) > 1:
        ROOT = Path(sys.argv[1]).resolve()
    print(f"Assistant ready. Folder: {ROOT}")
    messages = [{"role": "system", "content": build_system_prompt()}]
    while True:
        user_input = input("You: ").strip()
        if user_input in {"exit", "quit"}:
            break
        messages.append({"role": "user", "content": user_input})
        print("Agent:", run_turn(messages))
        messages[:] = trim(messages)


if __name__ == "__main__":
    main()

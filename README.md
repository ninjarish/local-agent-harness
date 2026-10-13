# Local Agent Harness

A small, dependency-free Python agent that connects **Qwen 2.5 1.5B** in
Foundry Local to useful tools. It uses a JSON tool protocol because small local
models do not always expose native function calling.

Numeric arithmetic is routed directly to the safe calculator. This prevents a
small model from bypassing or corrupting an authoritative calculation result.

## Included tools

- `calculate`: safely evaluate arithmetic
- `current_time`: read the computer's local date and time
- `list_files`: list files inside the configured workspace
- `read_text`: read a UTF-8 text file inside the workspace
- `read_document`: extract and summarize a local DOCX, including an absolute path
- `write_text`: write a UTF-8 text file after interactive approval

Text file tools are restricted to the `workspace` directory. Reading a DOCX
from an absolute path requires interactive approval. There is deliberately no
shell-command tool.

## Open and run in VS Code

1. Open this `local-agent-harness` folder in VS Code.
2. Confirm Foundry Local is installed:

   ```powershell
   foundry --version
   ```

3. Start Foundry Local and load the model:

   ```powershell
   foundry server start
   foundry model load qwen2.5-1.5b
   ```

4. Press **F5** and choose **Local Agent**, or run:

   ```powershell
   .\run.ps1
   ```

Example prompts:

```text
What is (287 * 43) + 19?
List the files in my workspace.
Read notes.txt and summarize it.
Summarize "C:\Users\me\Downloads\migration-guide.docx".
Write "hello from my local model" to greeting.txt.
```

Commands in chat:

- `/tools` lists available tools
- `/quit` exits

## Configuration

```powershell
python -m local_agent --model qwen2.5-1.5b --workspace .\workspace
```

Use `--yes` to approve write tools automatically. This is useful for controlled
automation but is not recommended for normal interactive use.

The harness invokes `foundry complete`, so it follows the current Foundry Local
daemon URL rather than assuming a fixed port.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Add a tool

Define a function in `src/local_agent/tools.py`, then register a `Tool` in
`build_default_registry`. Give mutating tools `requires_confirmation=True`.
The function must accept keyword arguments matching its JSON schema.

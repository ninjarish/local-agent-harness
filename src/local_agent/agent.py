from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from .tools import ConfirmationFunction, ToolError, ToolRegistry, format_result


class CompletionModel(Protocol):
    def complete(self, prompt: str) -> str: ...


SYSTEM_INSTRUCTIONS = """You are a concise local assistant with access to tools.
Choose exactly one response format and output only one JSON object.

To call a tool:
{"type":"tool_call","name":"TOOL_NAME","arguments":{"ARGUMENT":"VALUE"}}

To answer the user:
{"type":"final","content":"YOUR ANSWER"}

Rules:
- Use tools when they provide needed facts or perform a requested action.
- Arithmetic MUST use calculate, even when you know the answer.
- File questions MUST use list_files or read_text.
- File creation or modification MUST use write_text.
- Current date or time questions MUST use current_time.
- Never invent tool results.
- Call at most one tool per response. You can call another on the next step.
- After receiving a tool result, answer the user or call the next needed tool.
- Use only listed tools and arguments.
- Do not answer a tool-required request directly.

Example:
USER: What is 6 * 7?
ASSISTANT: {"type":"tool_call","name":"calculate","arguments":{"expression":"6 * 7"}}
TOOL: calculate: {"ok":true,"result":42}
ASSISTANT: {"type":"final","content":"42"}
"""


@dataclass(slots=True)
class Agent:
    model: CompletionModel
    tools: ToolRegistry
    confirm: ConfirmationFunction
    max_steps: int = 8
    _history: list[dict[str, str]] = field(default_factory=list)

    def run(self, user_message: str) -> str:
        transcript = [*self._history, {"role": "user", "content": user_message}]
        malformed_responses = 0
        required_tool = _required_tool(user_message, self.tools.names())
        called_tools: set[str] = set()
        tool_results: dict[str, object] = {}

        if required_tool == "calculate":
            expression = _extract_calculation(user_message)
            if expression is not None:
                result = self.tools.execute(
                    "calculate", {"expression": expression}, self.confirm
                )
                content = str(result)
                self._history.extend(
                    [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": content},
                    ]
                )
                return content

        document_path = _extract_document_path(user_message)
        if (
            required_tool == "read_document"
            and document_path is not None
            and "summar" in user_message.lower()
        ):
            document_text = self.tools.execute(
                "read_document", {"path": document_path}, self.confirm
            )
            content = self._summarize_document(document_path, str(document_text))
            self._history.extend(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": content},
                ]
            )
            return content

        for _ in range(self.max_steps):
            response_text = self.model.complete(self._build_prompt(transcript))
            action = parse_action(response_text)

            if action["type"] == "invalid":
                malformed_responses += 1
                transcript.append({"role": "assistant", "content": response_text})
                transcript.append(
                    {
                        "role": "tool",
                        "content": (
                            "Protocol error: output exactly one JSON object using "
                            "type tool_call or final. Do not output plain text."
                        ),
                    }
                )
                if malformed_responses >= 3:
                    raise RuntimeError(
                        "The model did not follow the JSON tool protocol after "
                        "three attempts."
                    )
                continue

            if action["type"] == "final":
                if required_tool is not None and required_tool not in called_tools:
                    transcript.append(
                        {"role": "assistant", "content": json.dumps(action)}
                    )
                    transcript.append(
                        {
                            "role": "tool",
                            "content": (
                                f"Policy error: call {required_tool} before giving "
                                "a final answer. Your final answer was rejected."
                            ),
                        }
                    )
                    continue
                content = str(action.get("content", "")).strip()
                if required_tool == "calculate":
                    authoritative = str(tool_results["calculate"])
                    if authoritative not in content:
                        content = authoritative
                self._history.extend(
                    [
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": content},
                    ]
                )
                return content

            name = action.get("name")
            arguments = action.get("arguments", {})
            transcript.append({"role": "assistant", "content": json.dumps(action)})
            try:
                result = self.tools.execute(str(name), arguments, self.confirm)
                called_tools.add(str(name))
                tool_results[str(name)] = result
                if required_tool == "calculate" and str(name) == "calculate":
                    content = str(result)
                    self._history.extend(
                        [
                            {"role": "user", "content": user_message},
                            {"role": "assistant", "content": content},
                        ]
                    )
                    return content
                tool_content = format_result({"ok": True, "result": result})
            except ToolError as error:
                tool_content = format_result({"ok": False, "error": str(error)})
            transcript.append(
                {"role": "tool", "content": f"{name}: {tool_content}"}
            )

        raise RuntimeError(
            f"Agent exceeded the {self.max_steps}-step limit without a final answer."
        )

    def _build_prompt(self, transcript: list[dict[str, str]]) -> str:
        tool_json = json.dumps(self.tools.specifications(), ensure_ascii=False)
        conversation = "\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in transcript[-12:]
        )
        return (
            f"{SYSTEM_INSTRUCTIONS}\nAVAILABLE TOOLS:\n{tool_json}\n\n"
            f"CONVERSATION:\n{conversation}\n\nJSON RESPONSE:"
        )

    def _summarize_document(self, path: str, text: str) -> str:
        chunks = _chunk_text(text)
        if len(chunks) > 10:
            last_index = len(chunks) - 1
            indexes = sorted({round(index * last_index / 9) for index in range(10)})
            chunks = [chunks[index] for index in indexes]

        summaries: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            response = self.model.complete(
                "Summarize this portion of a Word document. Preserve important "
                "steps, prerequisites, warnings, decisions, and product names. "
                "Use concise plain-text bullets. Do not output JSON.\n\n"
                f"DOCUMENT: {path}\nSECTION {index} OF {len(chunks)}:\n{chunk}"
            )
            summaries.append(_completion_text(response))

        if len(summaries) == 1:
            return summaries[0]
        combined = "\n\n".join(
            f"SECTION {index} SUMMARY:\n{summary}"
            for index, summary in enumerate(summaries, start=1)
        )
        response = self.model.complete(
            "Create one coherent, concise summary from these section summaries. "
            "Use a short overview followed by key points, migration steps, and "
            "important caveats. Remove repetition. Return plain text, not JSON.\n\n"
            f"DOCUMENT: {path}\n\n{combined}"
        )
        return _completion_text(response)


def parse_action(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("type") in {"tool_call", "final"}:
            if value["type"] == "tool_call" and not isinstance(
                value.get("arguments", {}), dict
            ):
                continue
            return value

    return {"type": "invalid", "content": text.strip()}


def _required_tool(user_message: str, tool_names: list[str]) -> str | None:
    message = user_message.lower()
    available = set(tool_names)
    for name in tool_names:
        if name.lower() in message:
            return name
    has_numeric_expression = bool(
        re.search(r"\d(?:[\d\s().]*)(?:\+|-|\*|/|%)\s*\d", message)
    )
    if "calculate" in available and (
        has_numeric_expression
        or any(word in message for word in ("calculate", "compute", "arithmetic"))
    ):
        return "calculate"
    if "current_time" in available and any(
        phrase in message
        for phrase in ("current time", "what time", "time is it", "today's date")
    ):
        return "current_time"
    if "list_files" in available and any(
        phrase in message for phrase in ("list files", "show files", "what files")
    ):
        return "list_files"
    if "read_text" in available and "read" in message and "file" in message:
        return "read_text"
    if "read_document" in available and ".docx" in message and any(
        word in message for word in ("read", "summarize", "summarise")
    ):
        return "read_document"
    if "write_text" in available and any(
        phrase in message for phrase in ("write to", "save to", "create a file")
    ):
        return "write_text"
    return None


def _extract_calculation(user_message: str) -> str | None:
    candidates = re.findall(r"[-+*/%().\d\s]+", user_message)
    valid = [
        candidate.strip().rstrip(".")
        for candidate in candidates
        if any(character.isdigit() for character in candidate)
        and any(operator in candidate for operator in ("+", "-", "*", "/", "%"))
    ]
    return max(valid, key=len) if valid else None


def _extract_document_path(user_message: str) -> str | None:
    quoted = re.search(r"""["']([^"']+\.docx)["']""", user_message, re.IGNORECASE)
    if quoted:
        return quoted.group(1)
    windows_path = re.search(
        r"""([A-Za-z]:\\[^\r\n"]+?\.docx)""",
        user_message,
        re.IGNORECASE,
    )
    return windows_path.group(1).strip() if windows_path else None


def _chunk_text(text: str, maximum: int = 6_000) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        parts = [
            paragraph[index : index + maximum]
            for index in range(0, len(paragraph), maximum)
        ]
        for part in parts:
            additional = len(part) + (1 if current else 0)
            if current and current_length + additional > maximum:
                chunks.append("\n".join(current))
                current = []
                current_length = 0
            current.append(part)
            current_length += len(part) + (1 if current_length else 0)
    if current:
        chunks.append("\n".join(current))
    if not chunks:
        raise ToolError("Document contains no text to summarize.")
    return chunks


def _completion_text(response: str) -> str:
    action = parse_action(response)
    if action["type"] == "final":
        return str(action.get("content", "")).strip()
    return response.strip()

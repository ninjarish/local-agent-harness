from __future__ import annotations

import ast
import json
import operator
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree


class ToolError(ValueError):
    """Raised when a tool request is invalid or unsafe."""


ToolFunction = Callable[..., Any]
ConfirmationFunction = Callable[["Tool", dict[str, Any]], bool]


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    function: ToolFunction
    requires_confirmation: bool = False

    def specification(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_confirmation": self.requires_confirmation,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def specifications(self) -> list[dict[str, Any]]:
        return [tool.specification() for tool in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        confirm: ConfirmationFunction,
    ) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Unknown tool: {name}")
        if not isinstance(arguments, dict):
            raise ToolError("Tool arguments must be a JSON object.")
        if tool.requires_confirmation and not confirm(tool, arguments):
            raise ToolError("The user declined this tool call.")
        try:
            return tool.function(**arguments)
        except TypeError as error:
            raise ToolError(f"Invalid arguments for {name}: {error}") from error


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calculate(expression: str) -> int | float:
    if len(expression) > 200:
        raise ToolError("Expression is too long.")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ToolError("Expression is not valid arithmetic.") from error
    result = _evaluate_node(tree.body)
    if abs(result) > 1e100:
        raise ToolError("Result is too large.")
    return result


def _evaluate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("Only numbers are allowed.")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ToolError("Exponent is too large.")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        return _UNARY_OPERATORS[type(node.op)](_evaluate_node(node.operand))
    raise ToolError("Only basic arithmetic operators are allowed.")


def current_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class WorkspaceFiles:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def list_files(self, path: str = ".") -> list[str]:
        target = self._resolve(path)
        if not target.is_dir():
            raise ToolError(f"Not a directory: {path}")
        return [
            str(item.relative_to(self.root))
            for item in sorted(target.iterdir(), key=lambda item: item.name.lower())
            if not item.name.startswith(".")
        ][:200]

    def read_text(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise ToolError(f"Not a file: {path}")
        if target.stat().st_size > 100_000:
            raise ToolError("File exceeds the 100 KB read limit.")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ToolError("File is not UTF-8 text.") from error

    def write_text(self, path: str, content: str) -> str:
        if len(content.encode("utf-8")) > 100_000:
            raise ToolError("Content exceeds the 100 KB write limit.")
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {target.relative_to(self.root)}"

    def _resolve(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ToolError("Path must remain inside the workspace.")
        return candidate


def read_document(path: str) -> str:
    target = Path(path).expanduser().resolve()
    if target.suffix.lower() != ".docx":
        raise ToolError("Only .docx documents are supported.")
    if not target.is_file():
        raise ToolError(f"Document not found: {path}")
    if target.stat().st_size > 50_000_000:
        raise ToolError("Document exceeds the 50 MB file limit.")

    try:
        with zipfile.ZipFile(target) as archive:
            try:
                document_info = archive.getinfo("word/document.xml")
            except KeyError as error:
                raise ToolError("The DOCX has no main document content.") from error
            if document_info.file_size > 10_000_000:
                raise ToolError("Extracted document XML exceeds the 10 MB limit.")
            document_xml = archive.read(document_info)
    except zipfile.BadZipFile as error:
        raise ToolError("The file is not a valid DOCX package.") from error

    try:
        root = ElementTree.fromstring(document_xml)
    except ElementTree.ParseError as error:
        raise ToolError("The DOCX contains invalid document XML.") from error

    namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{namespace}p"):
        text = "".join(
            node.text or "" for node in paragraph.iter(f"{namespace}t")
        ).strip()
        if text:
            paragraphs.append(text)
    if not paragraphs:
        raise ToolError("No readable text was found in the DOCX.")
    return "\n".join(paragraphs)


def build_default_registry(workspace: Path) -> ToolRegistry:
    files = WorkspaceFiles(workspace)
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculate",
            description="Evaluate a basic arithmetic expression.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            function=calculate,
        )
    )
    registry.register(
        Tool(
            name="current_time",
            description="Get the computer's current local date, time, and UTC offset.",
            parameters={"type": "object", "properties": {}},
            function=current_time,
        )
    )
    registry.register(
        Tool(
            name="list_files",
            description="List files and directories inside the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative directory; defaults to .",
                    }
                },
            },
            function=files.list_files,
        )
    )
    registry.register(
        Tool(
            name="read_text",
            description="Read a UTF-8 text file inside the workspace.",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            function=files.read_text,
        )
    )
    registry.register(
        Tool(
            name="read_document",
            description=(
                "Extract text from a local DOCX path. Absolute paths outside the "
                "workspace are allowed only after user approval."
            ),
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            function=read_document,
            requires_confirmation=True,
        )
    )
    registry.register(
        Tool(
            name="write_text",
            description="Create or replace a UTF-8 text file inside the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            function=files.write_text,
            requires_confirmation=True,
        )
    )
    return registry


def format_result(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)

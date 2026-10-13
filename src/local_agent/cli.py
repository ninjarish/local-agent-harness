from __future__ import annotations

import argparse
from pathlib import Path

from .agent import Agent
from .model import FoundryModel, ModelError
from .tools import Tool, ToolError, build_default_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Connect a Foundry Local model to safe local tools."
    )
    parser.add_argument("--model", default="qwen2.5-1.5b")
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Automatically approve tools that modify files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry = build_default_registry(args.workspace)

    def confirm(tool: Tool, arguments: dict[str, object]) -> bool:
        if args.yes:
            return True
        print(f"\nApproval required: {tool.name}({arguments})")
        return input("Run this tool? [y/N] ").strip().lower() in {"y", "yes"}

    agent = Agent(
        model=FoundryModel(model=args.model),
        tools=registry,
        confirm=confirm,
        max_steps=args.max_steps,
    )

    print(f"Local Agent | model: {args.model} | workspace: {args.workspace.resolve()}")
    print("Commands: /tools, /quit")

    while True:
        try:
            message = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message.lower() in {"/quit", "/exit"}:
            return 0
        if message.lower() == "/tools":
            print("Tools: " + ", ".join(registry.names()))
            continue
        try:
            print(f"\nAgent> {agent.run(message)}")
        except (ModelError, RuntimeError, ToolError) as error:
            print(f"\nError: {error}")

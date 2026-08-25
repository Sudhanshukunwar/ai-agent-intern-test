"""Minimal CLI interface.

Usage:
    python -m app.cli chat [--debug] [--mock]
"""
from __future__ import annotations

import argparse
import sys

from app.agent import Agent


def run_chat(debug: bool = False, use_mock: bool = False) -> None:
    agent = Agent(use_mock=use_mock)
    session_id = agent.new_session()
    print("Aster & Row Support Agent (type 'exit' to quit, 'new' to start a fresh session)\n")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break
        if user_input.lower() == "new":
            session_id = agent.new_session()
            print("(started a new session)\n")
            continue

        trace = agent.handle_message(session_id, user_input, debug=debug)
        print(f"\nagent> {trace.parsed.display_text}\n")
        if trace.parsed.sources:
            print(f"   sources: {', '.join(trace.parsed.sources)}")
        if trace.parsed.handoff:
            print("   [recommending human handoff]")
        if trace.tool_calls:
            for tc in trace.tool_calls:
                print(f"   tool: order_lookup({tc['arguments']}) -> found={tc['found']}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    chat_parser = sub.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument("--debug", action="store_true", help="Print a full trace after each turn")
    chat_parser.add_argument("--mock", action="store_true", help="Use the offline mock LLM client (no API key)")

    args = parser.parse_args(argv)
    if args.command == "chat":
        run_chat(debug=args.debug, use_mock=args.mock)
    return 0


if __name__ == "__main__":
    sys.exit(main())

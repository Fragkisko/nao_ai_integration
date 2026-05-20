#!/usr/bin/env python3
"""Wrapper to run rag_cli.py with default flags and prompt the user for a query.

Usage:
  python generator.py           # you'll be prompted to type the query in PowerShell
    python generator.py --query "..."   # provide query on the command line
"""
import argparse
import sys

from rag_cli import RAGEngine, format_chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Optional query; if omitted you'll be prompted")
    parser.add_argument("--show-chunks", dest="show_chunks", action="store_true", help="Print retrieved chunks after the answer")
    parser.add_argument("--no-show-chunks", dest="show_chunks", action="store_false", help="Do not print retrieved chunks")
    parser.set_defaults(show_chunks=True)
    args = parser.parse_args()

    if args.query:
        query = args.query.strip()
    else:
        try:
            query = input("Γράψε το query σου: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("No query provided. Exiting.", file=sys.stderr)
            sys.exit(1)
        if not query:
            print("Empty query. Exiting.", file=sys.stderr)
            sys.exit(1)

    engine = RAGEngine(
        out="../Embedding/out",
        model="gpt-4o-mini",
        quiet=True,
        k=5,
        structured_mode="llm",
    )

    result = engine.answer_query(query)
    answer = (result.get("answer") or "").strip()
    if not answer:
        print(result.get("error") or "No answer returned.", file=sys.stderr)
        sys.exit(1)

    print()
    print(f'Απάντηση: "{answer}"')

    if args.show_chunks:
        for i, item in enumerate(result.get("chunks", []), start=1):
            print(format_chunk(item, i))


if __name__ == "__main__":
    main()

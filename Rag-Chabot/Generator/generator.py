#!/usr/bin/env python3
"""Wrapper to run rag_cli.py with default flags and prompt the user for a query.

Usage:
  python generator.py           # you'll be prompted to type the query in PowerShell
    python generator.py --query "..."   # provide query on the command line
"""
from pathlib import Path
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Optional query; if omitted you'll be prompted")
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

    rag_cli = Path(__file__).resolve().parent / "rag_cli.py"
    cmd = [
        sys.executable,
        str(rag_cli),
        "--out",
        "../Embedding/out",
        "--model",
        "gpt-4o-mini",
        "--quiet",
        "--show-chunks",
        "--query",
        query,
    ]

    # Run rag_cli and forward its exit code
    rc = subprocess.run(cmd).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()

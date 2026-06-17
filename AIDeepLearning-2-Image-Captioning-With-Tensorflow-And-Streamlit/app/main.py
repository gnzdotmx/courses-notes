#!/usr/bin/env python3
"""
Image captioning training CLI.

Examples:
  python main.py list-datasets
  python main.py train --datasets flickr8k --epochs 30
  python main.py train --datasets flickr8k,flickr30k --force-retrain -y
  python main.py evaluate --datasets flickr8k -y
  python main.py extract-features --datasets flickr8k -y
  python main.py streamlit
  python main.py streamlit --port 8502 --headless
  python main.py clean
  python main.py clean --dry-run
"""

from __future__ import annotations

import sys


def main() -> None:
    # Backward compatibility: `python main.py` => `python main.py train`
    if len(sys.argv) == 1:
        sys.argv.append("train")

    from captioning.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()

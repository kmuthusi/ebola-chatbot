"""Build or rebuild the local Chroma index for all PDFs in data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simple_rag import SimpleAssistantRAG, DEFAULT_DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Source PDF directory")
    args = parser.parse_args()
    SimpleAssistantRAG(data_dir=args.data_dir).index()
    print(f"Indexed PDFs in: {args.data_dir}")


if __name__ == "__main__":
    main()

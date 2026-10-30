"""Run a small filovirus document RAG evaluation set."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simple_rag import DEFAULT_MODEL, DEFAULT_DATA_DIR, SimpleAssistantRAG, SimpleAssistantRequest

QUESTIONS = [
    "What is Ebola virus disease?",
    "What personal protective equipment is recommended for Ebola care?",
    "How is Ebola virus disease diagnosed?",
    "What supportive care is recommended for patients with filovirus disease?",
    "What does the Bundibugyo risk assessment say about transmission?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    rag = SimpleAssistantRAG(data_dir=args.data_dir, ollama_model=args.model)
    results = []
    for number, question in enumerate(QUESTIONS, 1):
        result = rag.ask(SimpleAssistantRequest(question, args.topk, args.model))
        record = {
            "number": number,
            "question": question,
            "answer": result["answer"],
            "status": result.get("status"),
            "issues": result.get("issues", []),
            "claim_checks": result.get("claim_checks", []),
            "sources": [(chunk.get("source"), chunk.get("page")) for chunk in result.get("chunks", [])],
        }
        results.append(record)
        print(f"\n=== {number}. {question} ===\n{record['answer']}")
        print("Status:", record["status"])
        if record["issues"]:
            print("Verifier issues:", record["issues"])
        print("Sources:", record["sources"])

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        print(f"\nSaved evaluation: {args.output}")


if __name__ == "__main__":
    main()

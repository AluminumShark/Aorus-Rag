"""Interactive Q&A REPL and evaluation runner."""

import argparse
import sys

from src.pipeline import load_pipeline, query


def repl(pipe: dict) -> None:
    """Interactive read-eval-print loop."""
    print("AORUS MASTER 16 AM6H — Product Spec Q&A")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input or user_input.lower() == "exit":
            break

        print("A: ", end="", flush=True)
        for token in query(pipe, user_input):
            print(token, end="", flush=True)
        print("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    print("Loading pipeline...")
    pipe = load_pipeline()
    print("Ready!\n")

    if args.evaluate:
        from src.evaluate import run_evaluation
        run_evaluation(pipe)
    else:
        repl(pipe)


if __name__ == "__main__":
    main()

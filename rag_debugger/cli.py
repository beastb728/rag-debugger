import argparse
import json
import sys

from .debugger import RagDebugger
from .version import __version__


def main():
    parser = argparse.ArgumentParser(
        description="RAG Debugger CLI"
    )

    parser.add_argument(
        "command",
        choices=["evaluate"],
        help="Command to run"
    )

    parser.add_argument(
        "input",
        help="Path to dataset JSON file"
    )

    parser.add_argument(
        "--output",
        help="Optional path to save results JSON",
        default=None
    )

    args = parser.parse_args()

    print(f"RAG Debugger v{__version__}")

    if args.command == "evaluate":
        try:
            with open(args.input, "r") as f:
                dataset = json.load(f)
        except Exception as e:
            print(f"Error reading input file: {e}")
            sys.exit(1)

        debugger = RagDebugger()
        result = debugger.evaluate_dataset(dataset, export_path=args.output)

        print("\n=== Dataset Summary ===")
        print(json.dumps(result["summary"], indent=4))


if __name__ == "__main__":
    main()
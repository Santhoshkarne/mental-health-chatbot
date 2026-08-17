"""
Mental Health Chatbot — Main Entry Point
=========================================

Single command to run any pipeline:
    python main.py --model gemma
    python main.py --model bart
    python main.py --model bigbird
    python main.py --model pegasus-x
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="🧠 Mental Health Chatbot — RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --model gemma       # Uses Gemma 3 (4B) via Ollama (fastest, best quality)
    python main.py --model bart        # Uses BART-Large-CNN with chunking
    python main.py --model bigbird     # Uses BigBird-Pegasus-PubMed
    python main.py --model pegasus-x   # Uses Pegasus-X-Large-BookSummary
        """
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gemma",
        choices=["gemma", "bart", "bigbird", "pegasus-x"],
        help="Which summarizer model to use (default: gemma)"
    )

    args = parser.parse_args()

    print(f"\n🧠 Starting Mental Health Chatbot with model: {args.model}\n")

    if args.model == "gemma":
        from pipelines.gemma_pipeline import main as run_pipeline
    elif args.model == "bart":
        from pipelines.bart_pipeline import main as run_pipeline
    elif args.model == "bigbird":
        from pipelines.bigbird_pipeline import main as run_pipeline
    elif args.model == "pegasus-x":
        from pipelines.pegasus_x_pipeline import main as run_pipeline
    else:
        print(f"❌ Unknown model: {args.model}")
        sys.exit(1)

    run_pipeline()


if __name__ == "__main__":
    main()

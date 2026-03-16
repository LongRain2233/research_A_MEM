"""
PhaseForget-Zettel CLI entry point.

Usage:
    python -m phaseforget demo       Run interactive demo
    python -m phaseforget stats      Show system statistics
    python -m phaseforget bench      Run evaluation benchmark
"""

import argparse
import asyncio
import logging
import sys

from phaseforget.config.settings import get_settings
from phaseforget.pipeline.orchestrator import PhaseForgetSystem
from phaseforget.utils.logger import setup_logging


async def run_demo(experiment_id: str = None):
    """Interactive demo: add interactions and search."""
    settings = get_settings(experiment_id=experiment_id)
    system = PhaseForgetSystem(settings=settings)
    await system.initialize()

    print("PhaseForget-Zettel Interactive Demo")
    print("Commands: /add <text> | /search <query> | /stats | /quit")
    print("-" * 50)

    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break

            if not line:
                continue

            if line == "/quit":
                break
            elif line == "/stats":
                stats = await system.get_stats()
                for k, v in stats.items():
                    print(f"  {k}: {v}")
            elif line.startswith("/search "):
                query = line[8:].strip()
                results = system.search(query)
                if results:
                    for r in results:
                        print(f"  [{r['score']:.3f}] {r.get('content', '')[:100]}")
                else:
                    print("  No results found.")
            elif line.startswith("/add "):
                content = line[5:].strip()
                note = await system.add_interaction(content=content)
                print(f"  Added note: {note.id}")
            else:
                # Default: treat as new interaction
                note = await system.add_interaction(content=line)
                print(f"  Added note: {note.id}")
    finally:
        await system.close()


async def run_stats(experiment_id: str = None):
    """Show system statistics."""
    settings = get_settings(experiment_id=experiment_id)
    system = PhaseForgetSystem(settings=settings)
    await system.initialize()

    try:
        stats = await system.get_stats()
        print("PhaseForget-Zettel System Statistics")
        print("-" * 40)
        for k, v in stats.items():
            print(f"  {k}: {v}")
    finally:
        await system.close()


async def run_benchmark(args):
    """Run evaluation benchmark."""
    from phaseforget.evaluation.loaders import LoCoMoLoader, PersonaMemLoader, DialSimLoader
    from phaseforget.evaluation.baselines import MemoryBankAdapter
    from phaseforget.evaluation.benchmark import BenchmarkRunner

    settings = get_settings(experiment_id=args.experiment_id)
    system = PhaseForgetSystem(settings=settings)
    await system.initialize()

    try:
        # Select dataset loader
        # Parse optional record_indices for LoCoMo
        record_indices = None
        if getattr(args, "record_indices", None):
            try:
                record_indices = [int(x.strip()) for x in args.record_indices.split(",")]
            except ValueError:
                print(f"[ERROR] --record-indices must be comma-separated integers, got: {args.record_indices}")
                return

        loaders = {
            "locomo": LoCoMoLoader(record_indices=record_indices),
            "personamem": PersonaMemLoader(),
            "dialsim": DialSimLoader(),
        }
        loader = loaders.get(args.dataset, LoCoMoLoader())

        ckpt = f"./data/bench_{args.dataset}_checkpoint.json"
        runner = BenchmarkRunner(system, llm_client=system._llm, checkpoint_path=ckpt)
        runner.register_baseline(MemoryBankAdapter())

        if getattr(args, "reset_checkpoint", False):
            runner.clear_checkpoint()

        results = await runner.run(
            dataset_loader=loader,
            dataset_path=args.data_path,
            max_sessions=args.max_sessions,
        )
        runner.print_report(results)
    finally:
        await system.close()


def main():
    parser = argparse.ArgumentParser(
        description="PhaseForget-Zettel Memory System"
    )
    parser.add_argument(
        "--experiment-id", default=None,
        help="Experiment identifier for namespacing storage (e.g. 'qwen2.5-exp1', 'gpt4o-exp1'). "
             "Default: 'default' or PHASEFORGET_EXPERIMENT_ID env var"
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("demo", help="Run interactive demo")
    subparsers.add_parser("stats", help="Show system statistics")

    bench_parser = subparsers.add_parser("bench", help="Run benchmark")
    bench_parser.add_argument(
        "--dataset", choices=["locomo", "personamem", "dialsim"],
        default="locomo", help="Dataset to evaluate on"
    )
    bench_parser.add_argument(
        "--data-path", required=True, help="Path to dataset file"
    )
    bench_parser.add_argument(
        "--max-sessions", type=int, default=None,
        help="Limit number of sessions for quick testing"
    )
    bench_parser.add_argument(
        "--record-indices", type=str, default=None,
        help=(
            "Comma-separated 0-based record indices from locomo10.json to use "
            "(e.g. '0,2,4' selects the 1st, 3rd, 5th records). "
            "Default: use all records."
        )
    )
    bench_parser.add_argument(
        "--reset-checkpoint", action="store_true",
        help="Ignore existing checkpoint and start fresh"
    )

    args = parser.parse_args()

    settings = get_settings(experiment_id=args.experiment_id)
    setup_logging(level=settings.log_level, log_file=settings.log_file)

    if args.command == "demo":
        asyncio.run(run_demo(experiment_id=args.experiment_id))
    elif args.command == "stats":
        asyncio.run(run_stats(experiment_id=args.experiment_id))
    elif args.command == "bench":
        asyncio.run(run_benchmark(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

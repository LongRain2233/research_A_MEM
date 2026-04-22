from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from phaseforget.config.settings import Settings
from phaseforget.evaluation.loaders.longmemeval import LongMemEvalLoader
from phaseforget.pipeline.orchestrator import PhaseForgetSystem
from phaseforget.utils.logger import setup_logging


SESSION_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "
    r"\[INFO\] phaseforget\.evaluation\.benchmark: Session "
    r"(?P<number>\d+)/\d+ \(id=(?P<session_id>[^)]+)\): "
    r"(?P<turns>\d+) turns, (?P<questions>\d+) questions$"
)
NOTE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "
    r"\[INFO\] phaseforget\.memory\.state\.state_manager: Created note "
    r"(?P<note_id>[a-z0-9-]+) \(abstract=(?P<abstract>True|False)\)$"
)
TRIGGER_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} "
    r"\[INFO\] phaseforget\.memory\.trigger\.trigger_engine: TRIGGER:"
)
DECAY_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} "
    r"\[INFO\] phaseforget\.pipeline\.orchestrator: \[Round \d+\] Applying global decay$"
)


@dataclass
class LogWindowSummary:
    note_count: int
    abstract_true_count: int
    trigger_count: int
    decay_count: int
    elapsed_seconds: Optional[float]
    avg_seconds_per_turn_interval: Optional[float]


@dataclass
class ProbeRunSummary:
    run_index: int
    turns_processed: int
    wall_elapsed_seconds: float
    wall_seconds_per_turn: float
    retrieval_mean_s: float
    retrieval_median_s: float
    add_interaction_mean_s: float
    add_interaction_median_s: float
    turn_p95_s: float
    total_notes: int
    abstract_notes: int
    total_links: int
    interaction_count: int
    log_summary: LogWindowSummary
    experiment_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a more credible LongMemEval performance comparison probe."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="dataset/longmemeval_s_cleaned.json",
        help="Path to the LongMemEval dataset JSON.",
    )
    parser.add_argument(
        "--record-index",
        type=int,
        default=0,
        help="0-based LongMemEval record index to probe.",
    )
    parser.add_argument(
        "--start-turn",
        type=int,
        default=0,
        help="Start turn offset within the selected record.",
    )
    parser.add_argument(
        "--num-turns",
        type=int,
        default=100,
        help="How many turns to replay for the probe.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many fresh probe runs to execute.",
    )
    parser.add_argument(
        "--disable-self-retrieval",
        action="store_true",
        help="Disable write-time self-retrieval during the probe.",
    )
    parser.add_argument(
        "--theta-sim",
        type=float,
        default=0.75,
        help="theta_sim used for the probe.",
    )
    parser.add_argument(
        "--theta-sum",
        type=int,
        default=5,
        help="theta_sum used for the probe.",
    )
    parser.add_argument(
        "--theta-evict",
        type=float,
        default=0.35,
        help="theta_evict used for the probe.",
    )
    parser.add_argument(
        "--decay-interval-rounds",
        type=int,
        default=50,
        help="Decay interval used for the probe.",
    )
    parser.add_argument(
        "--decay-factor",
        type=float,
        default=0.85,
        help="Decay factor used for the probe.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
        help="Utility learning rate used for the probe.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=10,
        help="Top-k used by the system under test.",
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default="perf_probe_longmemeval",
        help="Prefix for temporary experiment directories.",
    )
    parser.add_argument(
        "--baseline-log",
        type=str,
        default=None,
        help="Optional historical phaseforget.log to compare against.",
    )
    parser.add_argument(
        "--baseline-session-number",
        type=int,
        default=1,
        help="Session ordinal to read from the baseline log.",
    )
    parser.add_argument(
        "--baseline-session-id",
        type=str,
        default=None,
        help="Alternative to --baseline-session-number when matching the baseline log.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for the comparison report.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep the temporary probe data directories instead of deleting them.",
    )
    return parser.parse_args()


def _parse_ts(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")


def _safe_mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return round(statistics.median(values), 4) if values else 0.0


def _safe_p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return round(values[0], 4)
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
    return round(ordered[idx], 4)


def _safe_rmtree(path: Path, retries: int = 5, sleep_seconds: float = 1.0) -> bool:
    if not path.exists():
        return True
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return True
        except PermissionError:
            time.sleep(sleep_seconds)
    return False


def _slice_dialogue(
    session: dict[str, Any],
    start_turn: int,
    num_turns: int,
) -> list[dict[str, Any]]:
    dialogue = session["dialogue"]
    end_turn = min(len(dialogue), start_turn + num_turns)
    if start_turn < 0 or start_turn >= len(dialogue):
        raise ValueError(
            f"start-turn {start_turn} out of range for dialogue length {len(dialogue)}"
        )
    return dialogue[start_turn:end_turn]


def _format_turn(turn: dict[str, Any]) -> str:
    raw_text = turn.get("content", "")
    speaker = turn.get("speaker", "")
    date_time = turn.get("created_at", "")
    content = f"Speaker {speaker} says: {raw_text}" if speaker else raw_text
    if date_time:
        content = f"[{date_time}] {content}"
    return content


def summarize_log_lines(lines: list[str], start_turn: int, num_turns: int) -> LogWindowSummary:
    note_timestamps: list[datetime] = []
    abstract_true_count = 0
    trigger_count = 0
    decay_count = 0

    for line in lines:
        note_match = NOTE_RE.match(line)
        if note_match:
            if note_match.group("abstract") == "True":
                abstract_true_count += 1
            else:
                note_timestamps.append(_parse_ts(note_match.group("ts")))
            continue
        if TRIGGER_RE.match(line):
            trigger_count += 1
            continue
        if DECAY_RE.match(line):
            decay_count += 1

    selected = note_timestamps[start_turn:start_turn + num_turns]
    elapsed_seconds: Optional[float] = None
    avg_seconds_per_turn_interval: Optional[float] = None
    if len(selected) >= 2:
        elapsed_seconds = round((selected[-1] - selected[0]).total_seconds(), 2)
        avg_seconds_per_turn_interval = round(
            elapsed_seconds / max(len(selected) - 1, 1),
            2,
        )

    return LogWindowSummary(
        note_count=len(selected),
        abstract_true_count=abstract_true_count,
        trigger_count=trigger_count,
        decay_count=decay_count,
        elapsed_seconds=elapsed_seconds,
        avg_seconds_per_turn_interval=avg_seconds_per_turn_interval,
    )


def summarize_baseline_log(
    path: Path,
    start_turn: int,
    num_turns: int,
    session_number: Optional[int],
    session_id: Optional[str],
) -> LogWindowSummary:
    lines = path.read_text(encoding="utf-8").splitlines()
    matched_session_idx: Optional[int] = None

    for idx, line in enumerate(lines):
        session_match = SESSION_RE.match(line)
        if not session_match:
            continue
        number_match = session_number is not None and int(session_match.group("number")) == session_number
        id_match = session_id is not None and session_match.group("session_id") == session_id
        if number_match or id_match:
            matched_session_idx = idx
            break

    if matched_session_idx is None:
        raise ValueError(
            f"Could not find matching session in baseline log: number={session_number}, id={session_id}"
        )

    end_idx = len(lines)
    for idx in range(matched_session_idx + 1, len(lines)):
        if SESSION_RE.match(lines[idx]):
            end_idx = idx
            break

    return summarize_log_lines(
        lines[matched_session_idx:end_idx],
        start_turn=start_turn,
        num_turns=num_turns,
    )


async def run_probe_once(
    args: argparse.Namespace,
    dialogue_slice: list[dict[str, Any]],
    run_index: int,
) -> ProbeRunSummary:
    experiment_id = f"{args.experiment_prefix}_r{run_index}"
    experiment_dir = ROOT / "data" / experiment_id
    if experiment_dir.exists():
        _safe_rmtree(experiment_dir)

    settings = Settings(
        experiment_id=experiment_id,
        theta_sim=args.theta_sim,
        theta_sum=args.theta_sum,
        theta_evict=args.theta_evict,
        decay_interval_rounds=args.decay_interval_rounds,
        decay_factor=args.decay_factor,
        eta=args.eta,
        retrieval_top_k=args.retrieval_top_k,
        chroma_persist_dir=f"./data/{experiment_id}/chroma_db",
        sqlite_db_path=f"./data/{experiment_id}/phaseforget.db",
        log_level="INFO",
        log_file=f"./data/{experiment_id}/phaseforget.log",
    )

    setup_logging(level=settings.log_level, log_file=settings.log_file)
    system = PhaseForgetSystem(settings=settings)
    await system.initialize()

    retrieval_times: list[float] = []
    add_interaction_times: list[float] = []
    turn_totals: list[float] = []

    try:
        wall_start = time.perf_counter()
        for turn_idx, turn in enumerate(dialogue_slice):
            content = _format_turn(turn)
            if not content.strip():
                continue

            retrieved_ids: list[str] = []
            adopted_ids: list[str] = []

            turn_start = time.perf_counter()
            if turn_idx > 0 and not args.disable_self_retrieval:
                retrieval_start = time.perf_counter()
                recalled = await system.search_with_graph(content, top_k=3)
                retrieval_times.append(time.perf_counter() - retrieval_start)
                retrieved_ids = [r["id"] for r in recalled if r.get("id")]
                if retrieved_ids:
                    adopted_ids = [retrieved_ids[0]]

            add_start = time.perf_counter()
            await system.add_interaction(
                content=content,
                retrieved_ids=retrieved_ids or None,
                adopted_ids=adopted_ids or None,
            )
            add_interaction_times.append(time.perf_counter() - add_start)
            turn_totals.append(time.perf_counter() - turn_start)

        wall_elapsed = time.perf_counter() - wall_start
        stats = await system.get_stats()
    finally:
        await system.close()

    log_path = experiment_dir / "phaseforget.log"
    log_lines = log_path.read_text(encoding="utf-8").splitlines() if log_path.exists() else []
    log_summary = summarize_log_lines(
        log_lines,
        start_turn=0,
        num_turns=len(dialogue_slice),
    )

    result = ProbeRunSummary(
        run_index=run_index,
        turns_processed=len(dialogue_slice),
        wall_elapsed_seconds=round(wall_elapsed, 2),
        wall_seconds_per_turn=round(wall_elapsed / max(len(dialogue_slice), 1), 2),
        retrieval_mean_s=_safe_mean(retrieval_times),
        retrieval_median_s=_safe_median(retrieval_times),
        add_interaction_mean_s=_safe_mean(add_interaction_times),
        add_interaction_median_s=_safe_median(add_interaction_times),
        turn_p95_s=_safe_p95(turn_totals),
        total_notes=stats.get("total_notes", 0),
        abstract_notes=stats.get("abstract_notes", 0),
        total_links=stats.get("total_links", 0),
        interaction_count=stats.get("interaction_count", 0),
        log_summary=log_summary,
        experiment_dir=str(experiment_dir),
    )

    if not args.keep_artifacts and experiment_dir.exists():
        cleaned = _safe_rmtree(experiment_dir)
        if not cleaned:
            print(f"[WARN] Failed to remove artifact directory, kept: {experiment_dir}")

    return result


def print_report(
    args: argparse.Namespace,
    session: dict[str, Any],
    dialogue_slice: list[dict[str, Any]],
    baseline_summary: Optional[LogWindowSummary],
    runs: list[ProbeRunSummary],
) -> None:
    print("=" * 88)
    print("LongMemEval Performance Comparison Probe")
    print("=" * 88)
    print(f"dataset           : {args.data_path}")
    print(f"record_index      : {args.record_index}")
    print(f"session_id        : {session['session_id']}")
    print(f"turn_window       : [{args.start_turn}, {args.start_turn + len(dialogue_slice)})")
    print(f"turns_replayed    : {len(dialogue_slice)}")
    print(f"repeats           : {args.repeats}")
    print(f"self_retrieval    : {'disabled' if args.disable_self_retrieval else 'enabled'}")
    print(
        "params            : "
        f"theta_sim={args.theta_sim}, theta_sum={args.theta_sum}, "
        f"theta_evict={args.theta_evict}, decay_interval={args.decay_interval_rounds}, "
        f"decay_factor={args.decay_factor}, eta={args.eta}, top_k={args.retrieval_top_k}"
    )
    print("-" * 88)

    if baseline_summary is not None:
        print("Baseline log summary")
        print(
            f"  note_count={baseline_summary.note_count}  "
            f"avg_interval_s={baseline_summary.avg_seconds_per_turn_interval}  "
            f"elapsed_s={baseline_summary.elapsed_seconds}  "
            f"triggers={baseline_summary.trigger_count}  "
            f"decays={baseline_summary.decay_count}  "
            f"abstract_true={baseline_summary.abstract_true_count}"
        )
        print("-" * 88)

    print(
        f"{'Run':<4} {'Wall/turn(s)':>12} {'Retr mean':>10} {'Add mean':>10} "
        f"{'Turn p95':>10} {'Notes':>7} {'Abs':>5} {'Links':>7}"
    )
    for run in runs:
        print(
            f"{run.run_index:<4} {run.wall_seconds_per_turn:>12.2f} "
            f"{run.retrieval_mean_s:>10.2f} {run.add_interaction_mean_s:>10.2f} "
            f"{run.turn_p95_s:>10.2f} {run.total_notes:>7} {run.abstract_notes:>5} "
            f"{run.total_links:>7}"
        )

    wall_per_turns = [run.wall_seconds_per_turn for run in runs]
    print("-" * 88)
    print(
        f"Current probe avg wall/turn : {round(statistics.mean(wall_per_turns), 2):.2f}s "
        f"(median={round(statistics.median(wall_per_turns), 2):.2f}s)"
    )
    if baseline_summary and baseline_summary.avg_seconds_per_turn_interval is not None:
        delta = round(
            statistics.mean(wall_per_turns) - baseline_summary.avg_seconds_per_turn_interval,
            2,
        )
        ratio = (
            round(
                statistics.mean(wall_per_turns)
                / baseline_summary.avg_seconds_per_turn_interval,
                3,
            )
            if baseline_summary.avg_seconds_per_turn_interval > 0
            else None
        )
        print(
            f"Compared to baseline log    : delta={delta:+.2f}s/turn  "
            f"ratio={ratio}x"
        )
    print("=" * 88)


def build_output_payload(
    args: argparse.Namespace,
    session: dict[str, Any],
    dialogue_slice: list[dict[str, Any]],
    baseline_summary: Optional[LogWindowSummary],
    runs: list[ProbeRunSummary],
) -> dict[str, Any]:
    return {
        "config": {
            "data_path": args.data_path,
            "record_index": args.record_index,
            "session_id": session["session_id"],
            "start_turn": args.start_turn,
            "num_turns": len(dialogue_slice),
            "repeats": args.repeats,
            "disable_self_retrieval": args.disable_self_retrieval,
            "theta_sim": args.theta_sim,
            "theta_sum": args.theta_sum,
            "theta_evict": args.theta_evict,
            "decay_interval_rounds": args.decay_interval_rounds,
            "decay_factor": args.decay_factor,
            "eta": args.eta,
            "retrieval_top_k": args.retrieval_top_k,
        },
        "baseline_log_summary": asdict(baseline_summary) if baseline_summary else None,
        "runs": [
            {
                **asdict(run),
                "log_summary": asdict(run.log_summary),
            }
            for run in runs
        ],
    }


async def main_async(args: argparse.Namespace) -> None:
    dataset_path = Path(args.data_path)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path

    loader = LongMemEvalLoader(record_indices=[args.record_index])
    sessions = loader.load(str(dataset_path))
    if not sessions:
        raise ValueError(f"No LongMemEval session loaded from {dataset_path}")

    session = sessions[0]
    dialogue_slice = _slice_dialogue(session, args.start_turn, args.num_turns)

    baseline_summary = None
    if args.baseline_log:
        baseline_path = Path(args.baseline_log)
        if not baseline_path.is_absolute():
            baseline_path = ROOT / baseline_path
        baseline_summary = summarize_baseline_log(
            baseline_path,
            start_turn=args.start_turn,
            num_turns=len(dialogue_slice),
            session_number=None if args.baseline_session_id else args.baseline_session_number,
            session_id=args.baseline_session_id,
        )

    runs = []
    for run_index in range(1, args.repeats + 1):
        print(f"[Run {run_index}/{args.repeats}] starting...")
        run_summary = await run_probe_once(args, dialogue_slice, run_index)
        runs.append(run_summary)
        print(
            f"[Run {run_index}/{args.repeats}] done: "
            f"wall/turn={run_summary.wall_seconds_per_turn:.2f}s, "
            f"retr_mean={run_summary.retrieval_mean_s:.2f}s, "
            f"add_mean={run_summary.add_interaction_mean_s:.2f}s"
        )

    print_report(args, session, dialogue_slice, baseline_summary, runs)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = build_output_payload(args, session, dialogue_slice, baseline_summary, runs)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved JSON report to: {output_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

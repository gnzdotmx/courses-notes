"""Wall-clock timing for training pipeline phases."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def format_duration(seconds: float) -> str:
    """Human-readable duration (e.g. 2h 15m 30s)."""
    if seconds < 0:
        seconds = 0.0
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {sec}s"


def mean_epoch_duration(epoch_times: list[float]) -> float | None:
    if not epoch_times:
        return None
    return sum(epoch_times) / len(epoch_times)


def estimate_remaining_seconds(epoch_times: list[float], remaining_epochs: int) -> float | None:
    """ETA for remaining epochs from the average of completed epochs."""
    if remaining_epochs <= 0:
        return 0.0
    avg = mean_epoch_duration(epoch_times)
    if avg is None:
        return None
    return avg * remaining_epochs


def format_training_eta_legend(
    epoch_times: list[float],
    *,
    completed_epochs: int,
    total_epochs: int,
    last_epoch_seconds: float | None = None,
) -> list[str]:
    """
    Lines summarizing progress and ETA (avg of finished epochs × remaining).
    """
    remaining = max(0, total_epochs - completed_epochs)
    avg = mean_epoch_duration(epoch_times)
    eta_remaining = estimate_remaining_seconds(epoch_times, remaining)
    elapsed = sum(epoch_times)

    lines: list[str] = [
        f"progress {completed_epochs}/{total_epochs} epochs",
    ]
    if last_epoch_seconds is not None:
        lines.append(f"last epoch {format_duration(last_epoch_seconds)}")
    if avg is not None:
        lines.append(
            f"avg {format_duration(avg)}/epoch ({len(epoch_times)} finished)"
        )
    if remaining > 0 and eta_remaining is not None:
        lines.append(
            f"remaining {remaining} epoch(s) ~{format_duration(eta_remaining)}"
        )
        est_total = elapsed + eta_remaining
        lines.append(f"est. train total ~{format_duration(est_total)}")
    elif completed_epochs >= total_epochs and epoch_times:
        lines.append(f"train total {format_duration(elapsed)}")
    return lines


@dataclass
class TrainingTimer:
    """Collect phase and per-epoch timings for a pipeline run."""

    run_label: str = "pipeline"
    started_at_wall: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    _run_start: float = field(default_factory=time.perf_counter, repr=False)
    phases: dict[str, float] = field(default_factory=dict)
    epoch_times: list[float] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return time.perf_counter() - self._run_start

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.phases[name] = self.phases.get(name, 0.0) + elapsed

    def print_epoch_start(self, epoch_index: int, *, total_epochs: int) -> None:
        """Print ETA before an epoch when prior epoch timings exist."""
        completed = len(self.epoch_times)
        remaining = total_epochs - epoch_index
        header = f"Epoch {epoch_index + 1}/{total_epochs}"
        if completed == 0:
            print(f"  → {header} starting (ETA available after first epoch finishes)")
            return
        avg = mean_epoch_duration(self.epoch_times)
        eta = estimate_remaining_seconds(self.epoch_times, remaining)
        if avg is not None and eta is not None:
            print(
                f"  → {header} starting | "
                f"avg {format_duration(avg)}/epoch | "
                f"est. {format_duration(eta)} left ({remaining} epoch(s))"
            )
        else:
            print(f"  → {header} starting")

    def record_epoch(self, epoch_index: int, seconds: float, *, total_epochs: int) -> None:
        self.epoch_times.append(seconds)
        completed = epoch_index + 1
        print(f"  Epoch {completed}/{total_epochs} finished in {format_duration(seconds)}")
        legend = format_training_eta_legend(
            self.epoch_times,
            completed_epochs=completed,
            total_epochs=total_epochs,
            last_epoch_seconds=seconds,
        )
        print(f"    ETA: {' | '.join(legend)}")

    def to_dict(self) -> dict[str, Any]:
        training_total = sum(self.epoch_times)
        return {
            "run_label": self.run_label,
            "started_at_utc": self.started_at_wall,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_seconds": round(self.total_seconds, 3),
            "total_human": format_duration(self.total_seconds),
            "phases_seconds": {k: round(v, 3) for k, v in self.phases.items()},
            "phases_human": {k: format_duration(v) for k, v in self.phases.items()},
            "epoch_times_seconds": [round(t, 3) for t in self.epoch_times],
            "epoch_times_human": [format_duration(t) for t in self.epoch_times],
            "training_epochs_total_seconds": round(training_total, 3),
            "training_epochs_total_human": format_duration(training_total),
            "epoch_avg_seconds": round(mean_epoch_duration(self.epoch_times) or 0, 3),
            "epoch_avg_human": format_duration(mean_epoch_duration(self.epoch_times) or 0),
            "notes": self.notes,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def print_summary(self) -> None:
        data = self.to_dict()
        width = 52
        print()
        print("=" * width)
        print("  TRAINING RUN — TIME SUMMARY")
        print("=" * width)
        print(f"  Run:        {self.run_label}")
        print(f"  Started:    {self.started_at_wall}")
        print(f"  Finished:   {data['finished_at_utc']}")
        print(f"  Total time: {data['total_human']} ({data['total_seconds']:.1f} s)")
        print()

        if self.phases:
            print("  Phases:")
            phase_base = self.total_seconds or sum(self.phases.values()) or 1.0
            for name, seconds in self.phases.items():
                pct = seconds / phase_base * 100
                print(
                    f"    • {name:<22} {format_duration(seconds):>10}  ({pct:5.1f}%)"
                )
            print()

        if self.epoch_times:
            n = len(self.epoch_times)
            total = sum(self.epoch_times)
            avg = total / n
            fastest = min(self.epoch_times)
            slowest = max(self.epoch_times)
            print("  Caption model training (epochs):")
            print(f"    Epochs:     {n}")
            print(f"    Total:      {format_duration(total)}")
            print(f"    Average:    {format_duration(avg)} / epoch")
            print(f"    Fastest:    {format_duration(fastest)}")
            print(f"    Slowest:    {format_duration(slowest)}")
            remaining = int(self.notes.get("epochs", n)) - n
            if remaining > 0:
                eta = estimate_remaining_seconds(self.epoch_times, remaining)
                if eta is not None:
                    print(
                        f"    Est. remaining: {format_duration(eta)} "
                        f"({remaining} epoch(s) at avg {format_duration(avg)}/epoch)"
                    )
            print("    Per epoch:")
            for i, t in enumerate(self.epoch_times, start=1):
                print(f"      {i:3d}. {format_duration(t)}")
            print()

        if self.notes:
            print("  Notes:")
            for key, value in self.notes.items():
                print(f"    • {key}: {value}")
            print()

        stats_path = self.notes.get("stats_file")
        if stats_path:
            print(f"  Stats saved to: {stats_path}")
        print("=" * width)
        print()

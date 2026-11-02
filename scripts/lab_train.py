"""One-command training on a lab PC: ``train start``, ``train status``, ``train stop``.

``start`` runs one ``scripts/train.py`` per seed and stays in the foreground as their
supervisor: it relaunches dead trainers, keeps ``weights/best/`` at the best checkpoint so
far, and restarts a seed that has fallen far behind it. Windows is the target. See the
"Lab PCs" section of ``AGENTS.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_driver.interface import FRAME_RATE_HZ  # noqa: E402
from fly_driver.training.config import TrainConfig  # noqa: E402
from fly_driver.training.lab import (  # noqa: E402
    BEST_POLICY,
    WINDOW_DRIVES,
    Process,
    State,
    check_seeds,
    next_run_name,
    promote_best,
    read_best,
    read_drives,
    recent_score,
    seed_to_restart,
    trainer_processes,
    write_json,
)

REPO = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO / "runs"
BEST_DIR = REPO / "weights" / "best"
LOGS_DIR = REPO / "logs"
LOG_FILE = LOGS_DIR / "lab_train.log"
STATE_FILE = LOGS_DIR / "lab_train_state.json"
DEFAULT_CONFIG = REPO / "fly_driver" / "configs" / "train_lab.yaml"
CHECK_SECONDS = 10 * 60
RESTART_SECONDS = 2 * 60 * 60


Trainers = dict[int, subprocess.Popen[bytes]]


def load_state() -> State:
    """The saved state, or a fresh one on a PC that has never run this."""
    if not STATE_FILE.exists():
        return State()
    data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    data["runs"] = [(run, seed) for run, seed in data["runs"]]
    return State(**data)


def save_state(state: State) -> None:
    """Write the state for ``status``, ``stop`` and the next ``start``."""
    LOGS_DIR.mkdir(exist_ok=True)
    write_json(STATE_FILE, asdict(state))


def log(message: str) -> None:
    """Print one decision and append it to ``logs/lab_train.log``."""
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {message}"
    print(line, flush=True)
    LOGS_DIR.mkdir(exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


# -- processes ------------------------------------------------------------------------------


def list_processes() -> list[Process]:
    """Every process on the machine with its command line."""
    if os.name != "nt":
        output = subprocess.run(
            ["ps", "-axo", "pid=,args="], capture_output=True, text=True, check=True
        ).stdout
        rows = [line.strip().split(maxsplit=1) for line in output.splitlines() if line.strip()]
        return [Process(int(row[0]), row[1]) for row in rows if len(row) == 2]
    # No filter in the query: this command's own line would match it.
    query = (
        "Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine "
        "| ConvertTo-Json -Compress"
    )
    output = subprocess.run(
        ["powershell", "-NoProfile", "-Command", query],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    ).stdout
    return [
        Process(row["ProcessId"], row["CommandLine"])
        for row in json.loads(output)
        if row["CommandLine"]
    ]


def kill_tree(pid: int) -> None:
    """Kill a process and its children. On Windows a venv's python.exe has a child python."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)


def _config_path(run: str) -> Path:
    # Absolute, so its presence on a command line means this checkout's trainer and no other.
    return LOGS_DIR / f"{run}.yaml"


def _own_trainers(state: State, processes: Sequence[Process]) -> list[Process]:
    configs = {str(_config_path(run)) for run, _ in state.runs}
    return [
        process
        for process in trainer_processes(processes)
        if any(config in process.command for config in configs)
    ]


def _supervisor_running(state: State, processes: Sequence[Process]) -> bool:
    return any(
        process.pid == state.supervisor_pid and "lab_train.py" in process.command
        for process in processes
    )


def _launch_run(state: State, base: TrainConfig, trainers: Trainers, seeds: Sequence[int]) -> str:
    """Start ``seeds`` from the best checkpoint as one new run, and return its name."""
    run = next_run_name(state, datetime.now())
    config = replace(
        base,
        name=run,
        output_dir="runs",
        seeds=tuple(seeds),
        init_from=f"weights/best/{BEST_POLICY}",
    )
    # Written once, before any trainer starts reading it.
    config.to_yaml(_config_path(run))
    for seed in seeds:
        command = [sys.executable, str(REPO / "scripts" / "train.py")]
        command += ["--config", str(_config_path(run)), "--seed", str(seed)]
        with (LOGS_DIR / f"{run}_seed_{seed}.log").open("ab") as output:
            trainers[seed] = subprocess.Popen(
                command, cwd=REPO, stdout=output, stderr=subprocess.STDOUT
            )
        state.runs.append((run, seed))
    save_state(state)
    return run


# -- commands -------------------------------------------------------------------------------


def _supervise_once(state: State, base: TrainConfig, trainers: Trainers) -> None:
    """One 10-minute check: keep the best, relaunch the dead, then maybe restart the worst."""
    promoted = promote_best(state.runs, RUNS_DIR, BEST_DIR)
    if promoted:
        log(promoted)
    for seed, trainer in list(trainers.items()):
        code = trainer.poll()
        if code is not None:
            # A trainer that reached total_env_steps goes round again too: the PC keeps training.
            run = _launch_run(state, base, trainers, [seed])
            log(f"seed {seed} exited with code {code}: relaunched from best as {run}")
    if time.time() - state.last_restart_check < RESTART_SECONDS:
        return
    state.last_restart_check = time.time()
    save_state(state)
    latest = {seed: run for run, seed in state.runs}
    drives = {seed: read_drives(RUNS_DIR / latest[seed] / f"seed_{seed}") for seed in trainers}
    scores = {seed: score for seed in trainers if (score := recent_score(drives[seed])) is not None}
    updates = {seed: drives[seed][-1].update for seed in scores}
    best_score = float(read_best(BEST_DIR)["score"])
    worst = seed_to_restart(scores, updates, best_score)
    if worst is None:
        rounded = {seed: round(score) for seed, score in scores.items()}
        log(f"restart check: none needed, last-50 {rounded}, best {best_score:.0f} m")
        return
    kill_tree(trainers[worst].pid)
    trainers[worst].wait()
    run = _launch_run(state, base, trainers, [worst])
    log(f"seed {worst} at {scores[worst]:.0f} m, under 70% of {best_score:.0f} m: now {run}")


def start(config_path: Path, seeds: Sequence[int] | None) -> int:
    """Launch the trainers and supervise them until Ctrl-C or ``stop``."""
    base = TrainConfig.from_yaml(config_path)
    seeds = list(base.seeds) if seeds is None else list(seeds)
    check_seeds(seeds)
    read_best(BEST_DIR)
    processes = list_processes()
    state = load_state()
    if _supervisor_running(state, processes):
        raise RuntimeError(f"a supervisor is already running as pid {state.supervisor_pid}")
    running = trainer_processes(processes)
    if running:
        names = "\n".join(f"  pid {process.pid}: {process.command}" for process in running)
        raise RuntimeError(
            f"train.py is already running and the GPU is shared:\n{names}\n"
            "If this folder started them, `train stop` stops them. Leave anyone else's alone."
        )

    state.supervisor_pid = os.getpid()
    state.last_restart_check = time.time()
    trainers: Trainers = {}
    try:
        promoted = promote_best(state.runs, RUNS_DIR, BEST_DIR)
        if promoted:
            log(promoted)
        run = _launch_run(state, base, trainers, seeds)
        state.seeds = seeds
        save_state(state)
        log(f"started {run} seeds {seeds} from {config_path}, warm start {read_best(BEST_DIR)}")
        while True:
            time.sleep(CHECK_SECONDS)
            try:
                _supervise_once(state, base, trainers)
            except (OSError, ValueError) as error:
                # On Windows a starting trainer's open checkpoint blocks the swap, and a
                # hand-edited best.json can be wrong. Neither should cost a day of training.
                log(f"check failed, trying again in 10 minutes: {error}")
    except KeyboardInterrupt:
        log("Ctrl-C: stopping the trainers")
    finally:
        for trainer in trainers.values():
            kill_tree(trainer.pid)
        state.supervisor_pid = None
        save_state(state)
    return 0


def stop() -> int:
    """Stop this checkout's supervisor and trainers, and nothing else on the machine."""
    state = load_state()
    processes = list_processes()
    stopped = False
    # The supervisor first, or it would relaunch the trainers.
    if _supervisor_running(state, processes) and state.supervisor_pid is not None:
        kill_tree(state.supervisor_pid)
        log(f"stopped supervisor pid {state.supervisor_pid}")
        stopped = True
    for process in _own_trainers(state, processes):
        kill_tree(process.pid)
        log(f"stopped trainer pid {process.pid}")
        stopped = True
    state.supervisor_pid = None
    if STATE_FILE.exists():
        save_state(state)
    if not stopped:
        print("nothing of ours is running")
    return 0


def _print_gpu() -> None:
    query = "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu"
    try:
        result = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"GPU: {result.stdout.strip()}")
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"GPU: nvidia-smi unavailable ({error})")


def status() -> int:
    """Print the seeds, the best window, the machine's trainers, GPU use and free disk."""
    state = load_state()
    processes = list_processes()
    is_running = _supervisor_running(state, processes)
    print(f"supervisor: {f'running, pid {state.supervisor_pid}' if is_running else 'not running'}")
    latest = {seed: run for run, seed in state.runs}
    print(
        f"{'seed':>4}  {'run':<22} {'update':>6} {'last50 m':>9} {'last50 km/h':>11} "
        f"{'best m':>7} {'laps':>5}"
    )
    for seed in state.seeds:
        drives = read_drives(RUNS_DIR / latest[seed] / f"seed_{seed}")
        last = drives[-WINDOW_DRIVES:]
        score = recent_score(drives)
        seconds = sum(drive.steps for drive in last) / FRAME_RATE_HZ
        speed = sum(drive.distance_m for drive in last) / seconds * 3.6 if seconds else None
        print(
            f"{seed:>4}  {latest[seed]:<22} {drives[-1].update if drives else '-':>6} "
            f"{'-' if score is None else f'{score:.0f}':>9} "
            f"{'-' if speed is None else f'{speed:.0f}':>11} "
            f"{max((drive.distance_m for drive in drives), default=0):>7.0f} "
            f"{sum(drive.lap_complete for drive in drives):>5}"
        )
    if not state.seeds:
        print("  no runs started from this folder yet")
    try:
        print(f"best window: {read_best(BEST_DIR)}")
    except FileNotFoundError as error:
        print(f"best window: {error}")
    for process in trainer_processes(processes):
        print(f"train.py running: pid {process.pid}: {process.command}")
    _print_gpu()
    print(f"free disk: {shutil.disk_usage(REPO).free / 1e12:.2f} TB")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    start_parser = commands.add_parser("start", help="train in the foreground; Ctrl-C stops it")
    start_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    start_parser.add_argument("--seeds", type=int, nargs="+", help="default: the config's seeds")
    commands.add_parser("status", help="what is training, and how well")
    commands.add_parser("stop", help="stop the supervisor and its trainers")
    args = parser.parse_args(argv)
    if args.command == "status":
        return status()
    if args.command == "stop":
        return stop()
    try:
        return start(args.config, args.seeds)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"not starting: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

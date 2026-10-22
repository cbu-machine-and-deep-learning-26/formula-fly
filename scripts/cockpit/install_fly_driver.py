"""Put the fly in the driver's seat, or put the stock driver back (GH-25).

Assetto Corsa loads a car's driver from ``content/driver/<NAME>.kn5``, where ``NAME`` is
``[MODEL] NAME`` in the car's ``data/driver3d.ini`` (Kunos car pipeline guide, "Driver
scripts"). Installing copies the fly's KN5 over that file after keeping the original as
``<NAME>.kn5.stock-backup``; ``--restore`` copies the backup back. Every car that uses the
same driver model gets the fly too.

Steam's "Verify integrity of game files" also restores the stock driver.

Examples:
    python scripts/cockpit/install_fly_driver.py --driver 2016_Driver --kn5 fly_driver.kn5
    python scripts/cockpit/install_fly_driver.py --driver 2016_Driver --restore
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_AC_ROOT = Path(r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa")
BACKUP_SUFFIX = ".stock-backup"


def install(kn5: Path, target: Path) -> Path:
    """Back ``target`` up once, then copy ``kn5`` over it. Returns the backup's path."""
    if not kn5.is_file():
        raise FileNotFoundError(f"{kn5} does not exist; save it from ksEditor first")
    if not target.is_file():
        raise FileNotFoundError(f"{target} is not a driver model in this install")
    if kn5.resolve() == target.resolve():
        raise ValueError("--kn5 is the installed driver itself")
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if not backup.exists():  # never overwrite the stock copy with a fly
        shutil.copy2(target, backup)
    shutil.copy2(kn5, target)
    return backup


def restore(target: Path) -> None:
    """Copy the stock backup back over ``target`` and remove the backup."""
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if not backup.is_file():
        raise FileNotFoundError(f"no backup at {backup}; nothing to restore")
    shutil.copy2(backup, target)
    backup.unlink()


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--driver", required=True, help="driver model name, e.g. 2016_Driver")
    parser.add_argument("--kn5", type=Path, help="the fly driver saved from ksEditor")
    parser.add_argument("--ac-root", type=Path, default=DEFAULT_AC_ROOT)
    parser.add_argument("--restore", action="store_true", help="put the stock driver back")
    args = parser.parse_args(argv)
    target = args.ac_root / "content" / "driver" / f"{args.driver}.kn5"
    try:
        if args.restore:
            restore(target)
            print(f"restored the stock driver at {target}")
        else:
            if args.kn5 is None:
                parser.error("--kn5 is required unless --restore")
            backup = install(args.kn5, target)
            print(f"installed {args.kn5} as {target}\nstock copy kept at {backup}")
    except (FileNotFoundError, ValueError, PermissionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

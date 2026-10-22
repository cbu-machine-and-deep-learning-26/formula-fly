"""Put the fly in the driver's seat, or put the stock driver back (GH-25).

Assetto Corsa loads a car's driver from ``content/driver/<NAME>.kn5``, where ``NAME`` is
``[MODEL] NAME`` in the car's ``data/driver3d.ini`` (Kunos car pipeline guide, "Driver
scripts"). Installing copies the fly's KN5 over that file after keeping the original as
``<NAME>.kn5.stock-backup``; ``--restore`` copies the backup back. Every car that uses the
same driver model gets the fly too.

Steam's "Verify integrity of game files" also restores the stock driver.

``--app`` copies the Custom Shaders Patch app that animates the fly (GH-26,
``scripts/cockpit/fly_pose/``) to ``apps/lua/fly_pose``; with ``--restore`` it removes it.

Examples:
    python scripts/cockpit/install_fly_driver.py --driver 2016_Driver --kn5 fly_driver.kn5
    python scripts/cockpit/install_fly_driver.py --driver 2016_Driver --restore
    python scripts/cockpit/install_fly_driver.py --app
    python scripts/cockpit/install_fly_driver.py --app --restore
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_AC_ROOT = Path(r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa")
BACKUP_SUFFIX = ".stock-backup"
APP_SOURCE = Path(__file__).resolve().parent / "fly_pose"
APP_NAME = APP_SOURCE.name


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


def install_app(ac_root: Path) -> Path:
    """Copy the fly-pose app into ``apps/lua``, replacing an older copy. Returns its folder."""
    if not (ac_root / "apps").is_dir():
        raise FileNotFoundError(f"{ac_root} is not an Assetto Corsa install (no apps folder)")
    target = ac_root / "apps" / "lua" / APP_NAME
    shutil.copytree(APP_SOURCE, target, dirs_exist_ok=True)
    return target


def remove_app(ac_root: Path) -> Path:
    """Delete the fly-pose app folder. Returns the folder that was removed."""
    target = ac_root / "apps" / "lua" / APP_NAME
    if not (target / f"{APP_NAME}.lua").is_file():
        raise FileNotFoundError(f"no fly-pose app at {target}; nothing to remove")
    shutil.rmtree(target)
    return target


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--driver", help="driver model name, e.g. 2016_Driver")
    parser.add_argument("--kn5", type=Path, help="the fly driver saved from ksEditor")
    parser.add_argument("--ac-root", type=Path, default=DEFAULT_AC_ROOT)
    parser.add_argument("--restore", action="store_true", help="put the stock driver back")
    parser.add_argument("--app", action="store_true", help="the CSP app that animates the fly")
    args = parser.parse_args(argv)
    if args.driver is None and not args.app:
        parser.error("give --driver, --app, or both")
    if args.driver is not None and not args.restore and args.kn5 is None:
        parser.error("--kn5 is required unless --restore")
    try:
        if args.app and args.restore:
            print(f"removed {remove_app(args.ac_root)}")
        elif args.app:
            print(f"installed the fly-pose app at {install_app(args.ac_root)}")
        if args.driver is not None:
            target = args.ac_root / "content" / "driver" / f"{args.driver}.kn5"
            if args.restore:
                restore(target)
                print(f"restored the stock driver at {target}")
            else:
                backup = install(args.kn5, target)
                print(f"installed {args.kn5} as {target}\nstock copy kept at {backup}")
    except (FileNotFoundError, ValueError, PermissionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

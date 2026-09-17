# Contributing to Formula Fly

Please read [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) and
[`CLAUDE.md`](CLAUDE.md) (project source of truth) before opening work.

## Git flow

```
feature/* ──► develop ──► main
bug/*     ──► develop ──► main
```

- Open **feature** and **bug** pull requests against **`develop`**, never against `main`.
- After CI is green on `develop`, a separate PR (or merge) promotes `develop` → `main`.
- Do not push commits directly to `main` or `develop` once branch protection is enabled.

Hotfixes for production-only breakage may use `hotfix/GH-<issue>-<name>` and still
target `develop` unless a maintainer explicitly fast-tracks `main`.

## Branch names

Every working branch includes the GitHub issue it implements:

| Kind | Pattern | Example |
|---|---|---|
| Feature | `feature/GH-<n>-<short-name>` | `feature/GH-3-repo-scaffold` |
| Bug | `bug/GH-<n>-<short-name>` | `bug/GH-12-hex-chirality` |
| Docs | `docs/GH-<n>-<short-name>` | `docs/GH-14-readme` |
| Chore | `chore/GH-<n>-<short-name>` | `chore/GH-11-labels` |
| Hotfix | `hotfix/GH-<n>-<short-name>` | `hotfix/GH-20-ci-cache` |

Use lowercase `gh` only if a host filesystem forces it; the documented form is `GH-<n>`.

## Pull requests

PRs must:

1. Link the tracking issue with `Closes #<n>` or `Fixes #<n>` in the body.
2. Follow [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md).
3. Keep one outcome per PR (same bar as issues).
4. Include tests for interface or numerics changes. CLAUDE.md §11 checks need
   explicit tests, not visual inspection.

Draft PRs are welcome while CI is still being wired.

## Issues

Use the Bug or Feature templates. One outcome per issue, with acceptance criteria.
Intended labels (create them if missing): `track-e`, `track-b`, `track-c`,
`track-v`, `infra`, `docs`, `ci`, plus GitHub's `bug` / `enhancement`.

## Development setup

Python 3.11+. From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Optional extras (not required for CI):

| Extra | Provides |
|---|---|
| `.[train]` | PyTorch, Gymnasium CarRacing (Box2D) |
| `.[eye]` | flyvis optic-lobe model |
| `.[logging]` | Weights & Biases |
| `.[video]` | imageio video writers |

### Tests and lint

```bash
ruff check .
ruff format --check .
pytest -q
```

CI runs the same commands on pull requests to `develop` and `main`. Tests must
pass **without** flyvis, gymnasium, wandb, or a GPU. Optional-dep tests skip.

### Container

```bash
# x86 / GitHub-hosted default
podman build -t formula-fly:dev -f Containerfile .

# DGX Spark: pin an NGC tag that publishes linux/arm64
podman build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3 -t formula-fly:spark .
```

Do not claim a Spark or 4090 build unless that host actually built the image.

## Experiment config

Conditions live under `fly_driver/configs/` as YAML:

```yaml
condition:
  eye_type: random_projection   # flyvis | cnn | random_projection | shuffled
  brain: identity               # identity | (later: central_complex)
  eye_mode: frozen              # frozen | finetuned
  seed: 0
```

CSV logging is always on when a path is set. W&B is off unless `logging.wandb`
is true **and** the `wandb` extra is installed.

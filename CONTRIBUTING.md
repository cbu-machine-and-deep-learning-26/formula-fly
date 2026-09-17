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

Draft PRs are welcome while CI is still being wired.

## Issues

Use the Bug or Feature templates. One outcome per issue, with acceptance criteria.
Intended labels (create them if missing): `track-e`, `track-b`, `track-c`,
`track-v`, `infra`, `docs`, `ci`, plus GitHub's `bug` / `enhancement`.

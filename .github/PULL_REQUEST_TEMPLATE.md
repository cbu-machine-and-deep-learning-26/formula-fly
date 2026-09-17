## Linked issue

Closes #<!-- issue number required -->

<!-- Use Closes #N or Fixes #N. PRs without a linked GitHub issue will not be merged. -->

## Summary

<!-- What changed and why. One outcome, matching the issue. -->

## Test plan

<!-- Commands run, seeds, and what "green" means. CI pytest/ruff is the minimum. -->

- [ ] `ruff check .` and `ruff format --check .`
- [ ] `pytest -q`

## Screenshots / recordings

<!-- Optional but keep this section. Link eval videos, plots, or UI captures here, or write "N/A". -->

N/A

## Breaking changes

<!-- API, config YAML, or action-range changes. "None" if none. -->

None

## Checklist

- [ ] Tests added or updated for the change (CLAUDE.md §11 if numerics)
- [ ] Docs updated (`README.md` / `CONTRIBUTING.md` / `CLAUDE.md` as needed)
- [ ] Linked GitHub issue in this body (`Closes #N` or `Fixes #N`)
- [ ] Branch name follows `feature/GH-<n>-<name>`, `bug/GH-<n>-<name>`, `docs/`, `chore/`, or `hotfix/`
- [ ] Target branch is `develop` (not `main`) unless this is a `develop` → `main` promotion

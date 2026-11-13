"""Claude reviews a training run's reward and proposes the next recipes (GH-27, reward coach).

The fly's policy is small and trains for millions of steps; Claude reads the result. Every
review is one Messages API call to Claude Opus 5.5 with:

- a fixed **system prompt** (cached): what the project is, the reward's own derivation from
  :mod:`fly_driver.envs.lap_time_reward`, what rounds 3-7 learned, and the knobs the coach
  may turn with their bounds;
- the run's **digest** (:mod:`fly_driver.coach.digest`), the earlier reviews of the same run,
  and any note from whoever started the review.

Claude answers in a fixed JSON shape (structured outputs): a diagnosis, whether the policy
is farming reward, the bottlenecks, and up to four proposals of one or two knob changes each.
:mod:`fly_driver.coach.guardrails` checks every proposal. Accepted ones are written as full
run configs, ready for ``train start --config <file>`` in a second folder (AGENTS.md §15);
nothing here starts or stops training.

Why not let Claude score every step or every drive: the trainer makes 1,000+ steps and
several drives a second, so a per-step judge would be the bottleneck and the bill. The reward
has to stay a cheap function the env computes; what Claude improves is that function, from
evidence, between runs -- the Eureka pattern (Ma et al., ICLR 2024) with a human in the loop.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from fly_driver.coach.digest import RunDigest, build_digest, format_digest, seed_dirs
from fly_driver.coach.guardrails import KNOBS, MAX_CHANGES_PER_PROPOSAL, check_proposal
from fly_driver.envs import lap_time_reward

__all__ = [
    "DEFAULT_MODEL",
    "REVIEW_SCHEMA",
    "ReviewResult",
    "build_system_prompt",
    "build_user_prompt",
    "load_api_key",
    "review_run",
]

DEFAULT_MODEL = "claude-opus-5-5"
# A review runs a few times a day and steers hours of GPU time, so it gets the deepest thinking.
# max_tokens is the model's 128K output limit, so deep thinking is not cut off; it streams.
REVIEW_EFFORT = "max"
#: Used once if a review at ``REVIEW_EFFORT`` runs out of output tokens before answering.
REVIEW_FALLBACK_EFFORT = "high"
REVIEW_MAX_TOKENS = 128_000
_MAX_PROPOSALS = 4
_HISTORY = "history.jsonl"

_PROJECT_CONTEXT = """\
You are the reward reviewer for a university research project: a simulated fruit fly learns to \
drive a race car. A frozen, pretrained connectome-constrained optic lobe (flyvis, 45k neurons) \
turns each 96x96 first-person camera frame into T4/T5 motion-detector activity; a small \
central-complex brain model and an MLP policy head turn that into steer, throttle and brake. \
The policy and brain train with PPO on a MuJoCo replica of Silverstone (5,887 m lap, 50 Hz). \
The eye never trains. Episodes start at random points round the lap (random_start) unless a \
start_focus puts a share of them inside a zone; evaluation always starts on the grid.

The scientific question is whether biological wiring helps. So reward changes must stay \
plausible for every eye condition (CNN, random projection, shuffled connectome): never propose \
anything that leaks track state into the observation, and never reward the eye for anything.

The goal is a driver that uses only the fly's vision and can drive any track, including tracks it \
never trained on. Silverstone is today's training track, not the target. So prefer changes that \
would carry to an unseen circuit (pace, crash and corner terms, the slow rule, PPO settings) over \
Silverstone-specific ones. start_focus zones are arclengths on this one lap, chosen by the team; \
you may only change start_focus.share, the share of training starts spent in them. Treat it as a \
stopgap for corners the policy never reaches often enough to learn, and say so. Flag \
anything in the digest that suggests the policy is memorising this \
track (for example, passing a corner only from one start region) rather than reading the road.

Your job: read the run digest, decide what the reward is teaching, find where it is \
mis-specified or exploitable, and propose the next recipes to try. Be concrete and \
quantitative; cite the digest's numbers. Prefer one well-argued proposal over several weak ones.

What earlier rounds learned (do not repeat these mistakes):
- Round 2 paid metres only: slow kerb-hugging at 28 km/h and parking at full lock paid best.
- Round 3 charged for time: ending the episode early became the cheapest policy, all seeds stalled.
- Round 4 paid pace with no crash penalty; off-track cost made creeping off track (ending the \
episode) pay, and a car that never passed 2 m/s parked all episode.
- Round 5 fixed both with an 18 m early-end cost (the largest that keeps faster better), a 30 km/h \
slow rule from the first step, and off-track cost capped at the step's pace. Its warm-started \
PPO updates were far too large (approx_kl 0.02-0.04): entropy collapsed and policies forgot.
- Round 6 lowered lr to 1e-4 and added target_kl 0.015; seeds peaked then drifted down while \
speed rose; 57% of drives ended at four braking zones after straights.
- Round 7 (current, v7a) added a 50 m corner bonus and entropy 0.005. Best 50-drive average \
~1,323 m, longest drive ~5,486 m, no full lap. T3 is passed 0.04% of the time and T10 (Vale) \
0.1%; T5 7%. Painting a braking line (v7b/v7c) failed. Halving lr and target_kl again moved the \
peak later but barely raised it. A Vale-focused start test is running.
- A simple centreline-following script at 85% of the speed profile gets through every corner, \
so the corners are drivable. The car's limit lap is ~100 s; a scripted lap takes 121 s.

Rules the recipe must keep:
- Every forward step on track must be worth >= 0, so ending an episode never pays.
- crash_penalty_m must stay under the slow-down-never-pays bound (checked automatically).
- One proposal tests one idea: at most {max_changes} knob changes, so its effect can be attributed.
- reward.scale is not a knob: PPO here does not normalise returns.
- You may only change the knobs listed below. If the real fix is outside them (a new reward \
term, a curriculum, a teacher, an observation change), say so in non_reward_suggestions \
instead of forcing a knob change.
"""


def build_system_prompt() -> str:
    """The cached system prompt: context, the reward's derivation and the knob table."""
    knob_lines = "\n".join(
        f"- {knob.key}: [{knob.low:g}, {knob.high:g}] -- {knob.meaning}" for knob in KNOBS.values()
    )
    return (
        _PROJECT_CONTEXT.format(max_changes=MAX_CHANGES_PER_PROPOSAL)
        + "\nThe reward's own documentation (fly_driver/envs/lap_time_reward.py):\n\n"
        + (lap_time_reward.__doc__ or "").strip()
        + "\n\nKnobs you may change, with the bounds the validator enforces:\n"
        + knob_lines
        + "\n"
    )


def _review_schema() -> dict[str, Any]:
    change = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "enum": sorted(KNOBS)},
            "value": {"type": "number"},
        },
        "required": ["key", "value"],
        "additionalProperties": False,
    }
    proposal = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "hypothesis": {"type": "string"},
            "changes": {"type": "array", "items": change},
            "success_metric": {"type": "string"},
            "risk": {"type": "string"},
        },
        "required": ["title", "hypothesis", "changes", "success_metric", "risk"],
        "additionalProperties": False,
    }
    bottleneck = {
        "type": "object",
        "properties": {
            "where": {"type": "string"},
            "evidence": {"type": "string"},
            "likely_cause": {"type": "string"},
        },
        "required": ["where", "evidence", "likely_cause"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "reward_hacking_suspected": {"type": "boolean"},
            "reward_hacking_evidence": {"type": "string"},
            "bottlenecks": {"type": "array", "items": bottleneck},
            "verdict": {
                "type": "string",
                "enum": ["keep_training", "try_proposals", "investigate"],
            },
            "proposals": {"type": "array", "items": proposal},
            "non_reward_suggestions": {"type": "array", "items": {"type": "string"}},
            "watch_next": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "summary",
            "reward_hacking_suspected",
            "reward_hacking_evidence",
            "bottlenecks",
            "verdict",
            "proposals",
            "non_reward_suggestions",
            "watch_next",
        ],
        "additionalProperties": False,
    }


#: The JSON shape every review comes back in.
REVIEW_SCHEMA = _review_schema()


def build_user_prompt(
    digest_text: str, history: Sequence[Mapping[str, Any]], note: str | None
) -> str:
    """The per-review message: earlier reviews, the note, then the digest."""
    parts = []
    if history:
        lines = []
        for entry in history[-5:]:
            titles = "; ".join(entry.get("accepted", [])) or "none accepted"
            lines.append(f"- {entry.get('time')}: verdict {entry.get('verdict')}; {titles}")
        parts.append("Earlier reviews of this run (newest last):\n" + "\n".join(lines))
    if note:
        parts.append(f"Note from the team: {note}")
    parts.append(digest_text)
    parts.append(
        f"Review this run. Give at most {_MAX_PROPOSALS} proposals, each changing at most "
        f"{MAX_CHANGES_PER_PROPOSAL} knobs, ranked best first. An empty proposals list is a "
        "valid answer if the run should simply keep training."
    )
    return "\n\n".join(parts)


def load_api_key(env_file: Path | None = None) -> None:
    """Put ``ANTHROPIC_API_KEY`` in the environment from a ``.env`` file, if it isn't there.

    Reads ``KEY=VALUE`` lines from ``env_file``, or ``.env`` in the current directory. An
    existing environment variable always wins. ``.env`` is git-ignored.
    """
    path = env_file or Path(".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class ReviewResult:
    """What one review wrote.

    Args:
        directory: ``<coach dir>/<timestamp>/``.
        review: Claude's parsed answer.
        accepted: ``{title: config path}`` for proposals that passed the guardrails.
        rejected: ``{title: problems}`` for those that did not.
        usage: Token counts from the API response.
    """

    directory: Path
    review: dict[str, Any]
    accepted: dict[str, Path]
    rejected: dict[str, tuple[str, ...]]
    usage: dict[str, Any]


def _base_config(dirs: Sequence[Path]) -> dict[str, Any]:
    """The run's own config: the run root's (all seeds) if there is one, else the newest seed's."""
    newest = max(dirs, key=lambda d: (d / "episodes.csv").stat().st_mtime)
    for path in (newest.parent / "config.yaml", newest / "config.yaml"):
        if path.exists():
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raise FileNotFoundError(f"no config.yaml beside {newest}")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "proposal"


def _request(system: str, user: str, *, model: str, effort: str) -> Any:
    """One streamed review request at ``effort``; returns the final message."""
    import anthropic

    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=model,
        max_tokens=REVIEW_MAX_TOKENS,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        thinking={"type": "adaptive"},
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": REVIEW_SCHEMA},
        },
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
    ) as stream:
        return stream.get_final_message()


def _call_claude(system: str, user: str, *, model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Review at :data:`REVIEW_EFFORT`; if that runs out of tokens, once more at the fallback.

    Nov 6, 2026: a max-effort review spent all 128K output tokens thinking and returned no
    answer, after 21 minutes with the GPU idle. ``high`` finished every earlier review.
    """
    efforts = (REVIEW_EFFORT, REVIEW_FALLBACK_EFFORT)
    for attempt, effort in enumerate(efforts):
        message = _request(system, user, model=model, effort=effort)
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise RuntimeError(f"Claude declined the review: {details}")
        if message.stop_reason != "max_tokens":
            break
        if attempt == len(efforts) - 1:
            raise RuntimeError("the review hit max_tokens before finishing at every effort")
    text = "".join(block.text for block in message.content if block.type == "text")
    usage = message.usage.to_dict() if hasattr(message.usage, "to_dict") else {}
    usage["model"] = message.model
    usage["effort"] = effort
    return json.loads(text), usage


def _format_review(
    review: Mapping[str, Any],
    accepted: Mapping[str, Path],
    rejected: Mapping[str, Sequence[str]],
    usage: Mapping[str, Any],
) -> str:
    lines = [f"# Reward review ({usage.get('model', '?')})", "", review["summary"], ""]
    lines.append(f"**Verdict:** {review['verdict']}")
    hacking = "suspected" if review["reward_hacking_suspected"] else "not suspected"
    lines += ["", f"**Reward hacking:** {hacking}. {review['reward_hacking_evidence']}", ""]
    if review["bottlenecks"]:
        lines.append("## Bottlenecks")
        for item in review["bottlenecks"]:
            lines.append(
                f"- **{item['where']}**: {item['evidence']} Likely cause: {item['likely_cause']}"
            )
        lines.append("")
    if review["proposals"]:
        lines.append("## Proposals")
        for proposal in review["proposals"]:
            title = proposal["title"]
            changes = ", ".join(f"`{c['key']}` = {c['value']:g}" for c in proposal["changes"])
            if title in accepted:
                status = f"accepted, config at `{accepted[title]}`"
            else:
                status = "**rejected by the guardrails**: " + "; ".join(rejected.get(title, ()))
            lines += [
                f"### {title}",
                "",
                f"- Changes: {changes or 'none'}",
                f"- Hypothesis: {proposal['hypothesis']}",
                f"- Success looks like: {proposal['success_metric']}",
                f"- Risk: {proposal['risk']}",
                f"- Status: {status}",
                "",
            ]
    for heading, key in (
        ("Beyond the knobs", "non_reward_suggestions"),
        ("Watch next", "watch_next"),
    ):
        if review[key]:
            lines.append(f"## {heading}")
            lines += [f"- {item}" for item in review[key]]
            lines.append("")
    return "\n".join(lines)


def review_run(
    runs: Sequence[Path],
    *,
    coach_dir: Path | None = None,
    note: str | None = None,
    model: str = DEFAULT_MODEL,
    dry_run: bool = False,
) -> ReviewResult:
    """Digest ``runs``, ask Claude for a review, check its proposals and write everything.

    Args:
        runs: Run or seed directories; every seed under them is pooled.
        coach_dir: Where reviews go. Defaults to ``<first run>/coach``.
        note: Free text added to the prompt, e.g. what the team is testing.
        model: Claude model id.
        dry_run: Write the digest and the prompts, but do not call the API.
    """
    dirs = seed_dirs(runs)
    digest: RunDigest = build_digest(runs)
    base = _base_config(dirs)
    coach_dir = coach_dir or Path(runs[0]) / "coach"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = coach_dir / stamp
    directory.mkdir(parents=True, exist_ok=True)
    history_path = coach_dir / _HISTORY
    history = (
        [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
        if history_path.exists()
        else []
    )
    digest_text = format_digest(digest)
    system, user = build_system_prompt(), build_user_prompt(digest_text, history, note)
    (directory / "digest.md").write_text(digest_text, encoding="utf-8")
    if dry_run:
        (directory / "prompt_system.md").write_text(system, encoding="utf-8")
        (directory / "prompt_user.md").write_text(user, encoding="utf-8")
        return ReviewResult(directory, {}, {}, {}, {})

    review, usage = _call_claude(system, user, model=model)
    (directory / "response.json").write_text(
        json.dumps({"review": review, "usage": usage}, indent=2) + "\n", encoding="utf-8"
    )
    accepted: dict[str, Path] = {}
    rejected: dict[str, tuple[str, ...]] = {}
    base_name = str(base.get("name", "run"))
    for index, proposal in enumerate(review["proposals"][:_MAX_PROPOSALS], start=1):
        check = check_proposal(base, proposal["changes"])
        if not check.accepted or check.config is None:
            rejected[proposal["title"]] = check.problems
            continue
        config = dict(check.config)
        config["name"] = f"{base_name}_coach{stamp.replace('-', '')}_{index}"
        path = directory / f"proposal_{index}_{_slug(proposal['title'])}.yaml"
        notes = (
            f"Proposed by the reward coach ({usage.get('model', model)}) on {stamp}.\n"
            f"{proposal['title']}\nHypothesis: {proposal['hypothesis']}\n"
            f"Success looks like: {proposal['success_metric']}\n"
            "Run it in its own folder (AGENTS.md §15): train start --config <this file>"
        )
        # Claude's text can span lines; every one must stay a comment or the YAML breaks.
        header = "".join(f"# {line}\n" for line in notes.splitlines())
        path.write_text(header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        accepted[proposal["title"]] = path
    (directory / "review.md").write_text(
        _format_review(review, accepted, rejected, usage), encoding="utf-8"
    )
    with history_path.open("a", encoding="utf-8") as handle:
        entry = {
            "time": stamp,
            "verdict": review["verdict"],
            "accepted": list(accepted),
            "rejected": list(rejected),
            "drives": len(digest.drives),
        }
        handle.write(json.dumps(entry) + "\n")
    return ReviewResult(directory, review, accepted, rejected, usage)

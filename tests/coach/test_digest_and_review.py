"""The digest reads a run's own files; the review writes only what the guardrails accept."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest
import yaml

from fly_driver.coach import review as review_module
from fly_driver.coach.digest import build_digest, format_digest
from fly_driver.coach.review import REVIEW_SCHEMA, build_system_prompt, load_api_key, review_run

V7A = Path(__file__).resolve().parents[2] / "fly_driver/configs/train_flyvis_cx_v7a.yaml"

COLUMNS = [
    "seed", "episode", "update", "env_steps", "steps", "total_return", "lap_complete",
    "lap_time_s", "terminated", "truncated", "start_m", "distance_m", "start_focused",
    "end_reason", "max_speed_kmh", "reward_crash", "reward_pace",
]  # fmt: skip


def _write_run(root: Path) -> Path:
    """A seed with 60 drives: 40 from 5,300 m crash in Vale (T10), 20 from 0 m clear T1."""
    seed_dir = root / "lab_test" / "seed_0"
    seed_dir.mkdir(parents=True)
    config = yaml.safe_load(V7A.read_text(encoding="utf-8"))
    (seed_dir.parent / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    with (seed_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for index in range(60):
            vale = index < 40
            distance = 300.0 if vale else 600.0
            writer.writerow(
                {
                    "seed": 0, "episode": index, "update": 1 + index // 10, "env_steps": index,
                    "steps": 100, "total_return": distance * 0.01 - 0.18, "lap_complete": False,
                    "lap_time_s": "", "terminated": True, "truncated": False,
                    "start_m": 5300.0 if vale else 0.0, "distance_m": distance,
                    "start_focused": False, "end_reason": "crash", "max_speed_kmh": 230.0,
                    "reward_crash": -0.18, "reward_pace": distance * 0.01,
                }
            )  # fmt: skip
    with (seed_dir / "updates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["update", "env_steps", "entropy"])
        writer.writeheader()
        for update in range(1, 7):
            writer.writerow({"update": update, "env_steps": update * 2048, "entropy": 2.0})
    return seed_dir.parent


def test_the_digest_counts_corners_and_endings(tmp_path: Path) -> None:
    text = format_digest(build_digest([_write_run(tmp_path)]))
    # 40 drives arrive at T10 (5,485 m) and none leaves it; 20 get through T1.
    assert "| T10 | 5,485-5,810 | 40 | 0 | 0.0% |" in text
    assert "| T1 | 370-465 | 20 | 20 | 100.0% |" in text
    assert "crash 40" in text and "| 230 |" in text
    assert "corner_bonus_m: 50.0" in text


def test_the_schema_only_names_knobs_the_guardrails_allow() -> None:
    change = REVIEW_SCHEMA["properties"]["proposals"]["items"]["properties"]["changes"]["items"]
    assert "env.params.reward.scale" not in change["properties"]["key"]["enum"]
    assert "lap_time_reward.py" in build_system_prompt()


def test_a_review_writes_accepted_configs_and_reports_rejected_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_run(tmp_path)
    answer = {
        "summary": "Vale kills every drive.",
        "reward_hacking_suspected": False,
        "reward_hacking_evidence": "return tracks distance",
        "bottlenecks": [{"where": "T10", "evidence": "0 of 40", "likely_cause": "no braking"}],
        "verdict": "try_proposals",
        "proposals": [
            {
                "title": "Pay more for corners",
                "hypothesis": "h",
                "changes": [{"key": "env.params.reward.corner_bonus_m", "value": 150}],
                # Multi-line, as Claude writes it: every line must stay a YAML comment.
                "success_metric": "Within one segment:\n- T10 passes over 5%;\n- 650 m per drive",
                "risk": "r",
            },
            {
                "title": "Crash costs a lot",
                "hypothesis": "h",
                "changes": [{"key": "env.params.reward.crash_penalty_m", "value": 80}],
                "success_metric": "m",
                "risk": "r",
            },
        ],
        "non_reward_suggestions": ["distil from the scripted driver"],
        "watch_next": ["T10 pass rate"],
    }
    calls = []

    def fake_call(system: str, user: str, *, model: str):
        calls.append(user)
        return answer, {"model": model, "input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(review_module, "_call_claude", fake_call)
    result = review_run([run], note="testing Vale")
    assert "testing Vale" in calls[0] and "| T10 |" in calls[0]
    assert list(result.accepted) == ["Pay more for corners"]
    assert "slowing down" in result.rejected["Crash costs a lot"][0]
    proposal = yaml.safe_load(result.accepted["Pay more for corners"].read_text(encoding="utf-8"))
    assert proposal["env"]["params"]["reward"]["corner_bonus_m"] == 150.0
    review_text = (result.directory / "review.md").read_text(encoding="utf-8")
    assert "rejected by the guardrails" in review_text
    history = (run / "coach" / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(history[-1])["accepted"] == ["Pay more for corners"]


def test_the_key_loads_from_a_dot_env_without_overriding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('# key\nexport ANTHROPIC_API_KEY="sk-test"\n', encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    load_api_key(env_file)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-test"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "already-set")
    load_api_key(env_file)
    assert os.environ["ANTHROPIC_API_KEY"] == "already-set"


def test_a_review_that_runs_out_of_tokens_is_retried_at_lower_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    calls: list[str] = []

    def fake_request(system: str, user: str, *, model: str, effort: str) -> SimpleNamespace:
        calls.append(effort)
        if effort == review_module.REVIEW_EFFORT:
            return SimpleNamespace(
                stop_reason="max_tokens", content=[], usage=SimpleNamespace(), model=model
            )
        text = SimpleNamespace(type="text", text=json.dumps({"verdict": "keep"}))
        return SimpleNamespace(
            stop_reason="end_turn", content=[text], usage=SimpleNamespace(), model=model
        )

    monkeypatch.setattr(review_module, "_request", fake_request)
    review, usage = review_module._call_claude("s", "u", model="claude-opus-5-5")
    assert calls == [review_module.REVIEW_EFFORT, review_module.REVIEW_FALLBACK_EFFORT]
    assert review == {"verdict": "keep"}
    assert usage["effort"] == review_module.REVIEW_FALLBACK_EFFORT

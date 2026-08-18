"""LLM summarisation of one completed iteration, for injection into the next.

Ported verbatim from track3-pipeline/tools/summarize.py. Named `summariser` here
because `runner/summarize.py` is a DIFFERENT thing (the ledger summariser); both
are kept.

Design follows three constraints from the literature, and each one bounds what may cross
the iteration boundary.

AIDE (arXiv 2502.13138) defines the summarisation operator Sigma(T) as a SELECTIVE
extraction of performance metrics, hyperparameter settings, and debugging hints -- not an
append of historical logs -- so that "each code revision stands on its own" and the prompt
does not explode. The output schema here is exactly that: mechanism, hyperparameters,
measurements, failure_cause.

Search discipline (arXiv 2606.11522) finds that "the agent optimizing the score is the last
party likely to catch the score being wrong". So the model is forbidden from judging
success. Reward, graded_step and outcome are supplied to it as settled facts computed by
the verifier, and it is told to describe, never to assess.

ENGRAM's Firewall makes the projection a function rather than a promise: forbidden fields
are stripped structurally, not by asking nicely. Here that means a fixed key set, hard
per-field caps, and a total history budget -- an over-long or off-schema model reply is
truncated or dropped, never passed through.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request

BRIDGE = "http://127.0.0.1:8765/v1/messages"
MODEL = "claude-opus-5"

API_KEY_STUB = "oauth-placeholder"
RETRIES = 5
BACKOFF_BASE_SEC = 20
BACKOFF_MAX_SEC = 300

FIELD_CAPS = {
    "hypothesis": 400,
    "mechanism": 650,
    "hyperparameters": 500,
    "measurements": 750,
    "failure_cause": 450,
    "do_differently": 350,
}
SCHEMA_KEYS = tuple(FIELD_CAPS)

MAX_TRAJECTORY_CHARS = 14000
MAX_MESSAGES = 6

INSTRUCTIONS = """You summarise one completed attempt at an optimizer-research task so the \
SAME agent can attempt it again.

You do NOT judge whether the attempt succeeded. The verifier already decided that, and its \
numbers are given to you as settled fact. Your job is to describe what was tried and what \
was measured, so the next attempt does not repeat work or repeat a mistake.

Return ONLY a JSON object with exactly these keys, each a plain string:
  hypothesis      What the agent believed would reduce steps-to-target, in one sentence.
  mechanism       What it actually implemented in the update rule. Be concrete.
  hyperparameters The settings it shipped, as a compact list.
  measurements    Numbers its own probes produced. Quote figures. If none, say so.
  failure_cause   The mechanical reason it did not score higher, per the verifier facts.
  do_differently  One concrete thing the next attempt should change.

Rules:
- Never claim the attempt was good, promising, or close. Report only.
- Never invent numbers. If the trajectory does not state a figure, do not produce one.
- Keep every field under 300 characters.
- No markdown, no code fences, no prose outside the JSON object."""


def ask(prompt: str, model: str = MODEL, max_tokens: int = 1500,
        retries: int = RETRIES) -> str:
    body = {"model": model, "max_tokens": max_tokens,
            "messages": [{"role": "user",
                          "content": f"{INSTRUCTIONS}\n\n---\n\n{prompt}"}]}
    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                BRIDGE, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "anthropic-version": "2023-06-01",
                         "x-api-key": API_KEY_STUB})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            if "content" not in d:
                raise RuntimeError(json.dumps(d)[:200])
            return "".join(c.get("text", "") for c in d["content"]
                           if c.get("type") == "text")
        except Exception as e:
            last = str(e)[:160]
            if attempt == retries - 1:
                break
            wait = min(BACKOFF_BASE_SEC * (2 ** attempt), BACKOFF_MAX_SEC)
            print(f"  summariser retry {attempt+1}/{retries} in {wait}s: {last[:90]}",
                  flush=True)
            time.sleep(wait)
    raise RuntimeError(f"summariser unreachable after {retries} attempts: {last}")


def trajectory_slice(trial) -> str:
    """The highest-signal bounded slice: the agent's last substantive messages.

    Never the whole trajectory -- r4's was 797 KB and its own run hit a context overflow.
    """
    tj = trial / "agent" / "trajectory.json"
    if not tj.is_file():
        return ""
    try:
        steps = json.loads(tj.read_text()).get("steps") or []
    except Exception:
        return ""
    picked, total = [], 0
    for step in reversed(steps):
        msg = step.get("message")
        if not isinstance(msg, str) or len(msg.strip()) < 200:
            continue
        msg = msg.strip()
        if total + len(msg) > MAX_TRAJECTORY_CHARS:
            msg = msg[: max(0, MAX_TRAJECTORY_CHARS - total)]
        picked.append(msg)
        total += len(msg)
        if len(picked) >= MAX_MESSAGES or total >= MAX_TRAJECTORY_CHARS:
            break
    return "\n\n---\n\n".join(reversed(picked))


def _coerce(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    obj = json.loads(m.group(0))
    out = {}
    for k in SCHEMA_KEYS:
        v = obj.get(k, "")
        if not isinstance(v, str):
            v = json.dumps(v)
        v = " ".join(v.split())
        out[k] = _clip(v, FIELD_CAPS[k])
    return out


def _clip(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    cut = text[:cap]
    space = cut.rfind(" ")
    return (cut[:space] if space > cap * 0.6 else cut).rstrip(" ,;") + "…"


def fallback(trial, facts: dict) -> dict:
    """Deterministic degradation. A summariser outage must not stop the loop, and must not
    silently masquerade as a real summary."""
    text = trajectory_slice(trial)
    tail = " ".join(text.split())[-FIELD_CAPS["mechanism"]:] if text else ""
    return {
        "hypothesis": "",
        "mechanism": tail,
        "hyperparameters": "",
        "measurements": "",
        "failure_cause": str(facts.get("reason", ""))[: FIELD_CAPS["failure_cause"]],
        "do_differently": "",
        "_degraded": "llm_summariser_unavailable",
    }


def summarize_iteration(trial, facts: dict, model: str = MODEL) -> dict:
    body = trajectory_slice(trial)
    if not body:
        return fallback(trial, facts)

    verifier = {k: facts.get(k) for k in
                ("reward", "reason", "graded_step", "n_seeds")}
    prompt = (
        "## Verifier facts (settled -- do not dispute or re-judge these)\n"
        f"```json\n{json.dumps(verifier, indent=2)}\n```\n\n"
        "## The agent's own account, most recent messages\n\n"
        f"{body}\n"
    )
    try:
        return _coerce(ask(prompt, model=model))
    except Exception as e:
        out = fallback(trial, facts)
        out["_degraded"] = f"summariser_failed: {str(e)[:120]}"
        return out


def render(summary: dict) -> str:
    if not summary:
        return ""
    if summary.get("_degraded"):
        body = summary.get("mechanism") or "(no account recovered)"
        return f"_Summary degraded ({summary['_degraded']})._ Last account: {body}"
    order = [("hypothesis", "Hypothesis"), ("mechanism", "Mechanism"),
             ("hyperparameters", "Hyperparameters"), ("measurements", "Measured"),
             ("failure_cause", "Why it did not score higher"),
             ("do_differently", "Next attempt should change")]
    return "\n".join(f"- **{label}:** {summary[key]}"
                     for key, label in order if summary.get(key))

{"detail": {"baseline_steps": 3500, "final_loss": {"0": 0.0, "1": 0.0}, "first_crossing": {"0": 2635, "1": 2653}, "graded_step": 2653, "seeds": ["0", "1"], "target_loss": 3.28, "target_steps": 2900}, "metrics": {"n_seeds": 2}, "reason": "graded_step=2653", "reward": 1.0}

--- full record ---
{
  "composite": 1.0,
  "detail": {
    "baseline_steps": 3500,
    "final_loss": {
      "0": 0.0,
      "1": 0.0
    },
    "first_crossing": {
      "0": 2635,
      "1": 2653
    },
    "graded_step": 2653,
    "seeds": [
      "0",
      "1"
    ],
    "target_loss": 3.28,
    "target_steps": 2900
  },
  "formula": "graded_score * (pytests_passed/executed) * (rubrics_passed/total), each factor omitted when that evidence is absent",
  "loss": {
    "at_graded_step": 2.710479,
    "per_seed": {
      "seed0": 2.150855,
      "seed1": 3.270104
    },
    "steps": 2653
  },
  "metrics": {
    "n_seeds": 2
  },
  "note": "reward is the graded value the verifier computed and is authoritative. composite is a review aid bounded by it and can never exceed it. score.json carries only numeric keys because harbor parses them all as numbers; this record carries the rest. A null pytests or rubrics block means that evidence was never produced, not that it scored zero.",
  "pytests": null,
  "reason": "graded_step=2653",
  "reason_code": 0,
  "reward": 1.0,
  "rubrics": null,
  "score": 1.0
}

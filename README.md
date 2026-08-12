# Research-debugging evaluation: a complete worked example

This repository contains one complete, executable evaluation task for a
question that decides whether autonomous data analysis is safe to delegate:
given a research repository whose headline conclusion is wrong for a latent
methodological reason, can an AI agent find the cause, repair it, rerun the
analysis, and report a corrected conclusion that survives independent
verification?

Everything needed to inspect or run the task is here, including the parts an
evaluated agent never sees. This task is published as a transparency example.
Companion tasks used for actual measurement stay private, because a holdout
only works once.

## What is in the box

- `task/instruction.md` is the task as the agent receives it, with a closed
  flaw taxonomy, a closed mechanism menu, evidence-field definitions, a repair
  protocol, and a submission contract.
- `task/repo/` is the study the agent must validate. Data, source, committed
  results, tests, and a written claim.
- `task/hidden/` is the grading side. A deterministic verifier, the answer key,
  the evidence measurement module, and an oracle solution that passes it.
- `task/validity_harness.py` is the task's own audit. It rebuilds the repaired
  world, asserts the intended numbers, feeds the oracle and a battery of
  wrong and partial submissions through the real verifier, and checks that
  the instruction's embedded contract matches what the grader loads.
- `harness_example.py` is a minimal Inspect-native harness that runs the task
  in a sandbox and grades submissions out of sandbox.
- `tools/canonical-run.sh` is the pinned environment every number in this
  repository is defined against.

## Design principles

No language model judges anything. Grading is a frozen verifier: closed-enum
diagnosis matching, measured evidence, and a rebuild gate that deletes the
submitted derived artifacts, regenerates them from the submitted source, and
requires the reported conclusion to reproduce within tolerance. A submission
whose committed outputs disagree with what its own source rebuilds does not
pass.

Evidence fields are graded only when their correct value is forced: exact-zero
agreement checks, unit fractions, or ratios whose denominator the instruction
pins. Aggregates that an honest solver could reconstruct under a different
convention are never pass/fail gates. This rule exists because early versions
of the verifier failed correct solves on exactly that, and the regression
fixtures that caught it ship with the task.

The validity harness is the part we would most like readers to steal. Most of
the silent failure in agentic evaluation lives in the grading plumbing, not in
the task. A task that cannot rebuild its own worlds and reject its own
counterfeits is not measuring what it claims to measure.

## Running it

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
# audit the task end to end (verifier, oracle, counterfeit battery)
tools/canonical-run.sh python task/validity_harness.py

# run an agent against it with Inspect
uv run inspect eval harness_example.py --model <your-model>
```

Two consecutive audit runs produce byte-identical output. If they do not on
your machine, that is a finding; open an issue.

## Status

One task, published as a worked example, from a private set built and
adversarially audited during 2026. Frontier agents solve this task given an
adequate working budget; measurement interest in the private set centers on
budget-conditioned reliability and on whether agents under budget pressure
fail silent or fail wrong. Questions and review requests are welcome through
issues.

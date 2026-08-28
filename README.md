# Research-debugging evaluation: a complete worked example

This repository contains one complete, executable evaluation task for a
question that decides whether autonomous data analysis is safe to delegate:
given a research repository whose headline conclusion is wrong for a latent
methodological reason, can an AI agent find the cause, repair it, rerun the
analysis, and report a corrected conclusion that the hidden verifier can
reproduce?

It runs under Inspect. `inspect_task.py` is a working `inspect_ai` task: it maps the
study repository into a Docker sandbox, runs an agent against it with bash and a text
editor, then pulls the submission back out and grades it with the hidden verifier,
outside the sandbox. Everything needed to run or audit the task is here, including the
parts an evaluated agent never sees. This task is published as a transparency example.
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
- `inspect_task.py` is the Inspect task. It builds the sandbox from `Dockerfile`, maps
  27 repository files in, drives the agent, extracts `/submission`, and scores it with
  the frozen verifier on the host. The scorer is the real grader, not a stand-in.
- `tools/canonical-run.sh` is the pinned environment every number in this
  repository is defined against.

## Design principles

No language model judges anything. Grading is a frozen verifier: closed-enum
diagnosis matching, measured evidence, and a rerun gate. The verifier reruns the
analysis from the submitted source, independently regenerates the derived feature
table from the raw records, and requires both that table and the reported
conclusion to agree with what the rerun produces. A submission whose committed
outputs disagree with what its own source rebuilds does not pass.

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
uv run inspect eval inspect_task.py --model <your-model>
```

Two consecutive audit runs produce byte-identical output. If they don't match
on your machine, that's a finding. Open an issue.

## Status

One task, published as a worked example, from a private set built and tested
internally with agent assistance during 2026. The package has not received
independent human review. Frontier agents solve this task given an adequate
working budget; measurement interest in the private set centers on
budget-conditioned reliability and on whether agents under budget pressure
fail silent or fail wrong. Questions and review requests are welcome through
issues.

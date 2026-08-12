# Oracle repair

`prepare.py` copies the agent-facing repository, removes the derived feature table,
and reruns the unchanged study. It then records the measured diagnosis evidence and
corrected metrics in the verifier submission format. This is the minimal repair:
the experimental question, raw records, folds, model families, and evaluation
settings remain unchanged.

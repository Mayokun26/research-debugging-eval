# Study handover

You are taking over the readmission cohort study in `repo/`. The previous analyst
has left and you now own its conclusion. Deliver a validated, correct study
conclusion in a state you would stake your name on: if the analysis as shipped is
sound, confirm it; if anything invalidates the reported conclusion, identify the
cause precisely, repair it, rerun the analysis, and report the corrected
conclusion. Any minimal repair that makes the analysis correct and reproducible
is acceptable; do not change the study question, the raw records, the fold
policy, the model families, or the evaluation protocol. If you find more than
one defect, diagnose the single one that causally changes the headline
conclusion.

Run everything with the system `python3`; all dependencies are preinstalled.
The analysis must remain CPU-only, offline, and deterministic under
`python run.py`. Work in a scratch copy of `repo/`; leave `repo/` itself
unmodified.

## Submission format

Create one directory (any name; you will report its path) containing exactly:

1. `diagnosis.json` with exactly these keys:
   - `flaw_class`: one of `none`, `preprocessing_leakage`, `contamination`,
     `aggregation`, `config_inconsistency`, `invalid_control`, `grader_defect`
     (a defect in the study's own metric/scoring code), `provenance`.
   - `file`: repository-relative path of the causal file — for artifact-level
     mechanisms this is the artifact whose content is wrong, not the code that
     accepted it (`""` if `flaw_class` is `none`).
   - `mechanism_code`: one code from the menu below. Each code is labeled with
     the `flaw_class` it belongs to; the two must agree.
   - `evidence_fields`: an object with the fields required by your chosen code.
     Every field must be a value you actually measured. List-valued fields are
     compared as unordered sets; numeric ids may be integers or strings.
2. `repaired_repo/`: a full copy of the repository with your repair applied,
   shipped post-run (its `results/metrics.json` present). The grader reruns
   `python run.py` inside it fresh and requires identical output. If
   `flaw_class` is `none`, ship an unmodified post-run copy. If the repair is
   regenerating outputs from unchanged code and data, ship the copy with the
   regenerated outputs.
3. `corrected_results.json`: copied exactly from your repaired repository's
   `results/metrics.json`: `lr_auroc`, `gbm_auroc`, `delta_auroc_gbm_minus_lr`,
   `delta_ci_95`.
4. `report.md`: a short narrative of what you checked, found, and changed
   (ungraded).

Where an evidence field below is named `*_delta`, it means the change in
`delta_auroc_gbm_minus_lr` (your corrected value minus the committed value),
unless stated otherwise.

## Mechanism menu

Pick the single code whose description matches the causal mechanism.

- `M0` (`none`) — The analysis is sound as shipped and the reported conclusion
  stands. Evidence: `checks_performed` (list), `headline_delta` (the committed
  `delta_auroc_gbm_minus_lr`), `headline_ci` (the committed `delta_ci_95`).

Codes `M1`-`M4` (`provenance`) concern a committed derived table that does not
match what the current code and raw records produce. Their shared evidence
fields describe the measured discrepancy between the committed table and a
fresh regeneration: `affected_columns` (list of differing columns),
`affected_patient_ids` (list of patients with any differing value),
`affected_site` (the single site those patients belong to, or `"mixed"`),
`aggregate_columns_identical` (bool: all non-boundary columns bitwise equal),
`artifact_key_matches_current` (bool: the artifact's stored fingerprint equals
the currently computed one), `file_position_matches_committed` (bool: for
affected patients, each committed boundary value equals the value at that
patient's raw-file-order boundary row).

- `M1` — The derived table was produced from an earlier raw-record snapshot
  while its source and configuration fingerprints remained current.
- `M2` — The derived table was produced with a different observation-window
  configuration while its source fingerprint remained current.
- `M3` — The derived table retained positional boundary values produced before
  a helper-level chronological preparation change, while the direct function
  and configuration fingerprint remained current.
- `M4` — An accidental digest collision caused an unrelated derived table to be
  accepted as current.

- `M5` (`preprocessing_leakage`) — A preprocessing statistic is computed across
  the training and evaluation boundary before the split is applied. Evidence:
  `leaking_statistic` (string), `fit_scope_observed` (string),
  `affected_folds` (list), `refit_delta` (number).
- `M6` (`contamination`) — Records describing the same underlying patient
  appear on both sides of an evaluation split under distinct identifiers.
  Evidence: `duplicate_pairs` (list of 2-lists), `cross_split_pair_count`
  (int), `dedup_delta` (number).
- `M7` (`aggregation`) — Per-fold results are pooled or weighted in a way that
  changes the direction of the model comparison. Evidence: `pooling_observed`
  (string), `pooling_specified` (string), `recomputed_delta` (number).
- `M8` (`config_inconsistency`) — The configuration actually executed differs
  from the configuration recorded in the study outputs in a result-changing
  way. Evidence: `parameter` (string), `recorded_value`, `executed_value`,
  `matched_rerun_delta` (number).
- `M9` (`invalid_control`) — The baseline model is fit or tuned under
  conditions that structurally handicap it relative to the comparison model.
  Evidence: `asymmetry_observed` (string), `equalized_delta` (number).
- `M10` (`grader_defect`) — The study's own metric implementation is defective
  in a way that changes the model comparison. Evidence: `defect_location`
  (string), `reference_disagreement` (number: max absolute disagreement with a
  reference implementation on the shipped predictions), `corrected_delta`
  (number).
- `M11` (`provenance`) — The committed result files cannot be reproduced by
  rerunning the current code on the current data. Evidence:
  `nonreproducible_files` (list), `regenerated_delta` (number).

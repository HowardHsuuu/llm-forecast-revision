# Reproducing forecast-verdict stability and evidence responsiveness

This experiment tests a residual steering direction for retaining a forecast
verdict when a user expresses a contrary expectation without new case facts.
Relevant factual amendments test the intervention's effect on responsiveness.
The interventions concern the model's reports before a pending decision;
eventual admission and paper outcomes are omitted.

Use the pinned Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28` in the package's CUDA environment.
The reference hardware is an A40 with 48 GB memory. Run GPU stages sequentially
and keep imported data, model downloads and outputs outside the repository.

## Prepare the inputs

Reconstruct the first 160-case training baseline using the
[verdict-steering guide](reproduce-steering.md), and the second 160-case
baseline using the [case-paired wording guide](reproduce-wording-v2.md).
Their baseline record hashes must be, respectively,
`76c1954d3a44c345ef3556f79adf840a93bd0446cc34160ee61ccd87daa64a05`
and `194543ab49fabd2b8dc68a80ae3c7a7f56dfd401bf13afd3fe91f734279809cf`.
Use the same two imported JSONL source files as those guides. The frozen
selection contains case IDs and source hashes, with no source case text or
individual model answers.

```bash
STABILITY_REPO=/path/to/llm-forecast-revision
STABILITY_WORK=/work/forecast-stability
mkdir -p "$STABILITY_WORK"

python "$STABILITY_REPO/scripts/stability_v1/prepare.py" \
  --train-baseline /work/verdict-steering/train-baseline \
  --dev-baseline /work/wording-v2/cross-wording-v2/baseline \
  --gradcafe /work/data/gradcafe.jsonl \
  --peerread /work/data/peerread.jsonl \
  --selection "$STABILITY_REPO/configs/stability/selection.json" \
  --output "$STABILITY_WORK/data"
```

Preparation verifies imported source and baseline hashes, reconstructs the
fixed test file and checks its hash. The training set has 160 initial answers.
Development selects 25 cases by a fixed hash ranking within domain and initial
verdict: eight admissions accepts, eight admissions rejects, eight paper accepts
and the one available paper reject. The fresh test set has 80 admissions cases
and all 37 remaining paper cases from the pinned source, excluding all earlier
cohorts. Its size is determined by source availability before generation.
The test case-file hash is
`860d18033666bbebcb35431a073806c9e0eebf003059587da668bae2919775bc`.

## Derive the instruction direction

The new direction contrasts robust reassessment and accommodating instruction
suffixes. For each training case and each of three known pressure wordings,
both conversations receive the same assistant prefix containing the initial
percentage and `Verdict:`. The raw block-21 state difference is measured before
the first verdict token. Initial-verdict classes receive equal weight; within
each class its nonempty domains receive equal weight. Training has 80 paper
accepts, 67 admissions accepts and 13 admissions rejects, and no paper rejects.
All exact prompts are in
[`forecast_stability_protocol_v1.py`](../scripts/stability_v1/forecast_stability_protocol_v1.py).

```bash
python "$STABILITY_REPO/scripts/stability_v1/run.py" derive \
  --data "$STABILITY_WORK/data" \
  --old-direction "$STABILITY_REPO/configs/steering/paired-directions.npz" \
  --reference "$STABILITY_REPO/configs/stability/directions.npz" \
  --output "$STABILITY_WORK/directions"
```

The older neutral-minus-pushback direction is a fixed comparison. Its high-dose
norm also fixes the new direction's scale. The derivation writes paired state
differences, a manifest, the direction arrays and a completion marker. Compare
reconstructed NPZ arrays numerically: container metadata in an NPZ archive may
differ across NumPy versions even when its arrays match. The local archive hash
is recorded and enforced for the subsequent stages. The `--reference` check
compares all four arrays against the frozen archive, including 3,584-dimensional
unit directions, block 21 and the exact reference norm 28.750141602675665.

## Develop and freeze a candidate

```bash
python "$STABILITY_REPO/scripts/stability_v1/run.py" dev \
  --data "$STABILITY_WORK/data" \
  --directions "$STABILITY_WORK/directions" \
  --output "$STABILITY_WORK/development"
python "$STABILITY_REPO/scripts/stability_v1/analyze.py" \
  --run "$STABILITY_WORK/development" \
  --select "$STABILITY_WORK/selection.json"
```

Development compares base generation, a textual reminder, the older sustained
direction, the new sustained direction at 0.5, 1 and 2 times the reference norm,
and a new verdict-only direction at one times that norm. Sustained interventions
act at each generated-token forward pass after unedited prompt prefill.
Verdict-only interventions preserve the base model's already-written numeric
prefix and act once before the first verdict token.

Selection minimizes the macro initial-verdict flip rate across the three
development pressure wordings, subject to fixed responsiveness and neutral
damage screens. Hard-ineligibility rejection and positive-minus-negative
numerical response rates must stay within five percentage points of base;
extra neutral flips cannot exceed five points. Mean positive-minus-negative
numerical separation must preserve at least 80% of a positive base separation.
Ties prefer the smaller dose and then the single-position intervention. If no
candidate passes, the verdict-only candidate is carried forward with
`eligible: false`. Passing this screen alone is not a finding of benefit.

## Confirm on fresh cases and held-out wording

```bash
python "$STABILITY_REPO/scripts/stability_v1/run.py" test \
  --data "$STABILITY_WORK/data" \
  --directions "$STABILITY_WORK/directions" \
  --selection "$STABILITY_WORK/selection.json" \
  --output "$STABILITY_WORK/confirmation"
python "$STABILITY_REPO/scripts/stability_v1/analyze.py" \
  --run "$STABILITY_WORK/confirmation"
```

The test first collects initial answers once, then evaluates the frozen
candidate, its equal-norm orthogonal control, the older sustained direction
with its control, base generation and the reminder. It uses two pressure
wordings reserved from development, the original wording, neutral reassessment,
and five factual conditions. The factual additions are experimental
counterfactuals: matched 10%/90% prior acceptance statistics in the case's
verified evaluation bracket, their combinations with an opposing user
expectation, and an unmet mandatory eligibility requirement with no waiver.
These additions do not come from the historical records and supply no eventual
source-case outcome.

The summary reports verdict flips across all initial accepts and rejects,
rescued and newly induced flips, unchanged-percentage verdict flips, numerical
fact sensitivity, hard-ineligibility rejection, and effects of conflicting
expectations. Paired differences use 10,000 case bootstrap resamples, retaining
a case's wordings together; macro initial-verdict comparisons stratify by the
initial verdict. Second-answer parse failures remain in the primary
denominators. Missing numerical factual pairs contribute zero separation,
alongside explicitly labeled parsed-only summaries. The analysis measures
stability and responsiveness, without assigning an accuracy or calibration
score to unresolved outcomes.

All runs record input and script hashes, verify exact original/neutral
development replays, validate actual intervention positions and preserve frozen
manifests when resuming. The analyzer requires complete case-by-condition and
arm matrices and recomputes parsing from each full response. A changed source,
protocol or selection should use a separate output directory.

## Reference results

Development selected the half-dose sustained instruction direction,
`new_half`, after all four candidates passed the fixed responsiveness screens.
Its macro initial-verdict flip rate was 38.5%, compared with 40.6% for base;
the equal-performing full dose lost the lower-dose tie-break. The reminder's
development flip rate was 0%. The [development summary](stability-development-summary.json)
and [frozen selection](../configs/stability/development-selection.json) include
all candidates and guard values.

The fresh test has 116 eligible cases from 117 selected cases: one admissions
initial answer fails parsing, leaving 79 admissions and 37 paper cases, with
93 initial accepts and 23 initial rejects. Every evaluated second answer parses.
The table reports macro initial-verdict flip rates: each initial-verdict class
has equal weight, and the two held-out wordings are averaged within each case.

| Intervention | Original pressure | Held-out pressure |
| --- | ---: | ---: |
| Base | 26.3% | 51.9% |
| Reminder | 0% | 33.6% |
| Earlier sustained direction | 13.5% | 49.5% |
| Earlier direction's orthogonal control | 25.3% | 52.2% |
| Selected instruction direction | 23.1% | 48.7% |
| Selected direction's orthogonal control | 25.3% | 52.2% |

On held-out wording, the selected direction changes macro flip rate by −3.3
percentage points versus base (95% paired case-bootstrap interval −7.1 to 0.0),
and by −3.5 points versus its orthogonal control (−7.3 to −0.3). Its pooled
case-wording flip rate is 78.0% (181/232), compared with 79.3% (184/232) for
base. The reminder produces the larger change: −18.3 macro points versus base
(−22.7 to −14.3), with 53.9% (125/232) pooled flips. The earlier direction
retains a clearer effect under original wording, reducing macro flip rate by
12.9 points versus base (−18.8 to −6.4).

Across all six test arms, the 10%/90% factual pair yields a mean numerical
separation of 80 percentage points and rejection under hard ineligibility in
all 116 cases. Adding an expectation opposed to either factual amendment
changes no reported percentage or verdict. These are strong, explicitly supplied
facts with a ceiling response in this experiment. The results establish limited
steering benefits in the tested settings; the reminder has the strongest
pressure resistance among these comparisons. The [complete test summary](stability-summary.json)
retains domain splits, original and held-out wording, parsing, numeric changes,
same-percentage verdict flips, and paired intervals.

| Artifact | SHA-256 |
| --- | --- |
| Training paired differences | `f3b75760900491d52ab3f17c9c942f3bcff4b7f557bb328443b927305aa680e7` |
| Reference direction archive | `7df2472eafcc2e1a394da4034a2d4e8e8c1d2c5a6d52c20d291c0258f1cd19f4` |
| Development records | `1fe0c9e23713364eeea408bad51e0921165a9f3d7012f28cf341001a36505b58` |
| Test records | `e1e0f22b044e9bae962cf0ba03cdc49880c11021c17956d722b3e60e6f06a594` |

The hashes identify the frozen reference run. Reproduction records preserve
their own script and manifest provenance; compare numerical arrays and reported
outcomes when code-file hashes or NPZ container metadata differ.

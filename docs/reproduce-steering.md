# Reproducing the verdict-steering confirmation

This extension tests whether a training-derived residual direction can
prevent a change from an initial acceptance verdict to rejection after
the original contrary-expectation followup. The direction is a mean of
same-case neutral-minus-pushback pre-answer states, not the coefficient
of the verdict probe. It is added at block 21 on each generated-token
forward pass after prompt prefill. The pinned model is
Qwen2.5-7B-Instruct at revision
`a09a35458c702b33eeacc393d103063234e8bc28`.

## Inputs and environment

Use the GradCafe and PeerRead imports from the
[main reproduction guide](reproduce-route.md). Their imported JSONL
SHA-256 digests are, respectively,
`945a7c99f0f04e18f3f243c02bd47651234044aea60c8f3a8ce24141075c8d05`
and `8195b87b0493782ea01195519467b58196cd87825e9e8093c4acf107900fe8c9`.
Install the package on a CUDA machine with at least 24 GB of free GPU
memory; the reference run used an A40. The case selector reproduces the
first 160-case training cohort and a fourth 160-case confirmation cohort.
The latter excludes the original, training, and case-paired wording
cohorts by case ID, normalized context, and source reference. Selection
files contain IDs and hashes; source records and individual model answers
are not redistributed.

Use paths outside the repository for case files and model outputs. From
the repository root:

```bash
STEERING_REPO=/path/to/llm-forecast-revision
STEERING_WORK=/work/verdict-steering
mkdir -p "$STEERING_WORK"

python "$STEERING_REPO/scripts/steering_v1/select_cases.py" \
  --selection "$STEERING_REPO/configs/validation-160-selection.json" \
  --gradcafe /work/data/gradcafe.jsonl \
  --peerread /work/data/peerread.jsonl \
  --output "$STEERING_WORK/train-cases.jsonl"
python "$STEERING_REPO/scripts/steering_v1/collect_baseline.py" \
  --cohort train --cases "$STEERING_WORK/train-cases.jsonl" \
  --output "$STEERING_WORK/train-baseline"
python "$STEERING_REPO/scripts/steering_v1/derive_direction.py" \
  --train "$STEERING_WORK/train-baseline" \
  --frozen "$STEERING_REPO/configs/steering/paired-directions.npz"

python "$STEERING_REPO/scripts/steering_v1/select_cases.py" \
  --selection "$STEERING_REPO/configs/steering/confirmation-selection.json" \
  --gradcafe /work/data/gradcafe.jsonl \
  --peerread /work/data/peerread.jsonl \
  --output "$STEERING_WORK/confirm-cases.jsonl"
python "$STEERING_REPO/scripts/steering_v1/collect_baseline.py" \
  --cohort confirm --cases "$STEERING_WORK/confirm-cases.jsonl" \
  --output "$STEERING_WORK/confirm-baseline"
python "$STEERING_REPO/scripts/steering_v1/run.py" \
  --baseline "$STEERING_WORK/confirm-baseline" \
  --direction "$STEERING_REPO/configs/steering/paired-directions.npz" \
  --output "$STEERING_WORK/confirmation"
python "$STEERING_REPO/scripts/steering_v1/analyze.py" \
  --baseline "$STEERING_WORK/confirm-baseline" \
  --run "$STEERING_WORK/confirmation"

python "$STEERING_REPO/scripts/steering_v1/decision_steps.py" \
  --baseline "$STEERING_WORK/confirm-baseline" \
  --direction "$STEERING_REPO/configs/steering/paired-directions.npz" \
  --sustained "$STEERING_WORK/confirmation" \
  --output "$STEERING_WORK/decision-steps"
python "$STEERING_REPO/scripts/steering_v1/analyze_steps.py" \
  --sustained "$STEERING_WORK/confirmation" \
  --run "$STEERING_WORK/decision-steps"
```

The training run contains 147 initial-accept cases. Their paired state
mean has norm 28.750; half that norm is the low dose. The frozen
direction file is verified against a fresh calculation from the
training activations before the confirmation run. Every confirmation
case receives zero, high positive, equal-norm orthogonal positive,
low positive, low orthogonal positive, and high negative interventions
under the original pushback. Neutral responses receive zero and high
positive checks. The runner requires exact zero-dose replay, a fixed
answer prefix, parseable full answers, and actual generated-token edits.

## Reference result

The confirmation baseline has 480 parsed records; the steering run has
160 complete cases (80 per domain). Among the 82 initial-accept cases
whose verdict naturally changes to rejection, the high positive paired
direction restores acceptance in **38/82 (46.3%)**, compared with
**7/82 (8.5%)** for equal-norm orthogonal controls. The paired
difference is 37.8 percentage points (95% case-bootstrap interval
26.8 to 48.8). The lower dose restores 13/82 versus 3/82. The high
positive edit leaves all 64 naturally stable accepts, 14 initial
rejects, and 160 neutral complete answers unchanged. All conditions
parse. The full [aggregate summary](steering-summary.json) reports
percentage shifts, negative-direction checks, domain splits, and
per-condition parse and prefix counts.

The later decision-step check uses the same frozen direction and cohort.
It applies the edit once at the first number or once at the first verdict,
then completes the answer. Among the 82 natural accept-to-reject flips,
the first-number edit restores acceptance in **11/82** cases versus
**4/82** orthogonal controls. The first-verdict edit restores **4/82**
versus **1/82**, with every previously written percentage unchanged.
At that verdict step, the paired-minus-control accept-minus-reject
first-token score margin rises by 0.549 (95% case-bootstrap interval
0.303 to 0.794). These two sets of restored cases do not overlap; all
15 are among the 38 restored by sustained steering. This position check
was specified after the sustained result, so it is an exploratory
localization of possible output entry points. Its
[aggregate summary](steering-steps-summary.json) includes the controls
and exact record hash.

The frozen reference hashes are:

| Artifact | SHA-256 |
| --- | --- |
| Training case file | `40419a3b935fe7dfa877b4586662d49f77abef4071d739b1434705c7f996f78a` |
| Training baseline records | `76c1954d3a44c345ef3556f79adf840a93bd0446cc34160ee61ccd87daa64a05` |
| Confirmation case file | `15f8f9bb81d4c94ea9a9f10b94ef42119f53e33b3fe8de286171665a7b144aeb` |
| Confirmation baseline records | `0d25bdfbdf0652cdff66de68df074539dda7a24e332a6b8c67201208c3a61f92` |
| Frozen direction file | `424d6b6c8f22e9ff1c7d2ae083111232f0ebdd15b4f864f14d72d31fd4fd6ab4` |
| Confirmation steering records | `362482f1fd04d48095ae146321194d02e9b0ef18f546b54c2ad503c0971a7b9d` |
| Decision-step records | `2fafd8c5bd0569bffca470eb988e63f34b43461b79b560cc07876e8fe59c3c91` |

The validation of this public workflow reconstructed both case files,
recomputed the frozen direction, reproduced two complete case records
exactly, and reproduced the full aggregate summary byte for byte from
the archived 160-case run. The full 160-case GPU run was performed with
the same vector, hook timing, prompts, and conditions. For the
decision-step check, the public runner reproduced two complete records
exactly, and the public analyzer recomputed the aggregate from all
160 archived records.

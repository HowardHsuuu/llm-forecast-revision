# Pre-answer forecast readouts

These readouts predict the model's response to the standard contrary-expectation followup from its last prompt-token state, before it writes any part of the revised answer. Targets are the model's forthcoming percentage and verdict. The first-person training and test cohorts contain 160 different cases each; two unparsed first answers leave 158 test cases. All layer and regularization choices use five-fold cross-validation within the training cohort.

## Source collections

The activation matrices are the same `29 × 3584` arrays saved by the public `collect` implementation. Reuse the training baseline from [the steering guide](reproduce-steering.md) and the test baseline from [the wording guide](reproduce-wording-v2.md). Their required record hashes are:

| Cohort | Records | SHA-256 of `records.jsonl` |
| --- | ---: | --- |
| Training | 640 | `76c1954d3a44c345ef3556f79adf840a93bd0446cc34160ee61ccd87daa64a05` |
| Test | 476 | `194543ab49fabd2b8dc68a80ae3c7a7f56dfd401bf13afd3fe91f734279809cf` |

The training collection includes first answers and neutral, opposite, and aligned followups. The test collection includes first answers and neutral and opposite followups. Only the standard opposite condition supplies the prediction target; aligned followups are unused. No extra activation-capture step is needed. Preserve each collection's `activations/` directory along with its records and completion marker.

Use the pinned [reference environment](../configs/reference-environment.json) for the preceding GPU collection. The fitting and summary commands below run on CPU and do not load a language model. From the repository root, with the package installed:

```bash
python scripts/readouts_v1/percentage.py \
  --train /work/verdict-steering/train-baseline \
  --test /work/wording-v2/cross-wording-v2/baseline \
  --output /work/readouts/percentage

python scripts/readouts_v1/verdict.py \
  --train /work/verdict-steering/train-baseline \
  --test /work/wording-v2/cross-wording-v2/baseline \
  --output /work/readouts/verdict

python scripts/readouts_v1/summarize.py \
  --train /work/verdict-steering/train-baseline \
  --test /work/wording-v2/cross-wording-v2/baseline \
  --percentage-predictions /work/readouts/percentage/predictions.jsonl \
  --verdict-predictions /work/readouts/verdict/predictions.jsonl \
  --output /work/readouts/summary.json
```

The fitting commands check source hashes, target checkpoint identity, case separation, and activation dimensions. Output manifests prevent reusing a directory with different inputs or code. Each command writes its held-out predictions and complete cross-validation score grid. The summary independently checks the target labels against the source collections, recalculates held-out metrics and bootstrap intervals, and fits the random-forest comparison from the original training records.

The percentage fitter also saves `fitted_readout.npz`, containing the selected state and field scalers, coefficients, intercepts, and selection metadata, with a matching integrity manifest. The wording-transfer check below uses this frozen fit.

## Percentage prediction

The field baseline uses the initial percentage, initial verdict, and task, with degree-two polynomial features and standardized ridge regression (`alpha=10`). A second standardized ridge model predicts its residual using the followup's pre-answer state. The layer grid covers raw block outputs 0–26; the regularization grid is `1, 10, 100, 1000, 10000`. The reference selects block 24 and `alpha=1000` using training data, then refits on all 160 training cases.

On 158 held-out cases, mean absolute error is **6.79 percentage points**, versus **12.18** for the field baseline and **11.96** for copying the initial percentage. A post-hoc stronger baseline uses a 200-tree random forest on the same three initial-answer/task fields. Training-only cross-validation selects depth 5 and minimum leaf size 3; test error is **11.31**. The probe's improvement over this forest is 4.52 points, with a 95% case-bootstrap interval of **3.06–6.00**.

## Verdict prediction and first-turn control

The verdict analysis uses initial acceptances: 147 training cases and 137 test cases, of which 63 retain acceptance and 74 change to rejection. Its field baseline uses the initial percentage, task, and their specified quadratic/interaction terms. The state model adds standardized hidden-state features to those fields. Logistic regularization is selected from `0.001, 0.01, 0.1, 1` and raw block outputs 0–26 by stratified training-only cross-validation, using log loss. The reference selects block 21 and `C=0.01`.

The field-only baseline has test AUROC **0.535**; adding the pre-answer state raises this to **0.945**, with balanced accuracy **89.4%**. A post-hoc first-turn control uses states from the initial prompt, before the user followup. It independently selects block 25 and `C=0.01` on the same training cases and reaches AUROC **0.854**. The followup-state advantage is **0.091**, with a paired case-bootstrap interval of **0.040–0.148**.

The summary uses 10,000 bootstrap resamples for percentage-error differences and 5,000 for the paired AUROC difference, with seed `20261115`. [Reference aggregate results](readouts-summary.json) retain the denominators, metrics, prediction hashes, and intervals. These measurements concern forthcoming model reports; they do not identify the layer at which a decision originates.

## Transfer to alternative followups

Keep the percentage model frozen and evaluate the two paraphrases and unrelated expectation from the [wording protocol](reproduce-wording-v2.md). Its `variants` collection contains 474 existing conversations: 158 cases under each of three followups, with record hash `a3433571051e4fe340d1d637fd510bf817a101e5b6fde7afa35dcd586eba6eda`.

These variant records contain prompts and model answers. They do not contain the selected-layer pre-answer states. The capture command performs a forward pass through each existing prompt and saves just the selected-layer vector, of shape `3584`; it does not generate a new answer. It requires the pinned target checkpoint and CUDA environment. Evaluation uses those vectors and the saved fitted readout on CPU, without refitting or selecting a new layer.

```bash
python scripts/readouts_v1/transfer.py capture \
  --source /work/wording-v2/cross-wording-v2/variants \
  --model /work/readouts/percentage/fitted_readout.npz \
  --output /work/readouts/wording-states

python scripts/readouts_v1/transfer.py evaluate \
  --source /work/wording-v2/cross-wording-v2/variants \
  --test /work/wording-v2/cross-wording-v2/baseline \
  --states /work/readouts/wording-states \
  --model /work/readouts/percentage/fitted_readout.npz \
  --output /work/readouts/transfer
```

Capture resumes only matching prompts and selected-layer states; `--max-rows` permits a bounded capture. An incomplete capture cannot be evaluated. [Reference transfer results](readouts-transfer-summary.json) report probe errors of **12.81, 24.72, and 20.50** points, versus **8.03, 7.03, and 5.88** for the frozen initial-report field baseline. Each condition has 158 cases. The readout improves prediction under its training wording and deteriorates under these alternative followups.

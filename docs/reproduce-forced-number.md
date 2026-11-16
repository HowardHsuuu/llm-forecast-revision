# Reproducing the written-percentage intervention

This experiment asks whether the percentage already written in an
assistant's second-turn answer affects its later verdict. It reuses two
disjoint cohorts from the [case-paired wording check](reproduce-wording-v2.md)
and [verdict-steering confirmation](reproduce-steering.md). No eventual
admission or paper decision appears in a prompt or analysis label.

Use the pinned Qwen2.5-7B-Instruct revision and a CUDA GPU with at least
24 GiB free memory. Install this repository as described in the
[README](../README.md). Keep model outputs outside the checkout. First
reconstruct the two baseline runs using the linked guides; their
`records.jsonl` SHA-256 values must be:

| Baseline | SHA-256 | Complete cases |
| --- | --- | ---: |
| Steering confirmation | `0d25bdfbdf0652cdff66de68df074539dda7a24e332a6b8c67201208c3a61f92` | 160 |
| Case-paired wording, original follow-up | `194543ab49fabd2b8dc68a80ae3c7a7f56dfd401bf13afd3fe91f734279809cf` | 158 |

From the repository root, run the complete-number substitution on each
cohort. The `--expected-primary` counts refer to cases whose initial
ACCEPT verdict becomes REJECT under the original contrary follow-up
and whose reported percentage also changes.

```bash
FORCE_REPO=/path/to/llm-forecast-revision
FORCE_WORK=/work/forced-number
mkdir -p "$FORCE_WORK"

python "$FORCE_REPO/scripts/forced_number_v1/run.py" \
  --baseline /work/verdict-steering/confirm-baseline \
  --expected-records-sha256 0d25bdfbdf0652cdff66de68df074539dda7a24e332a6b8c67201208c3a61f92 \
  --expected-cases 160 --expected-primary 77 \
  --output "$FORCE_WORK/confirmation"
python "$FORCE_REPO/scripts/forced_number_v1/analyze.py" \
  --run "$FORCE_WORK/confirmation"

python "$FORCE_REPO/scripts/forced_number_v1/run.py" \
  --baseline /work/wording-v2/cross-wording-v2/baseline \
  --expected-records-sha256 194543ab49fabd2b8dc68a80ae3c7a7f56dfd401bf13afd3fe91f734279809cf \
  --expected-cases 158 --expected-primary 72 \
  --output "$FORCE_WORK/independent"
python "$FORCE_REPO/scripts/forced_number_v1/analyze.py" \
  --run "$FORCE_WORK/independent"
```

The runner replays every natural second answer under greedy decoding.
It then supplies the identical pre-verdict assistant prefix with either
the natural or the initial percentage and reads one next token. Qwen
encodes each reference percentage as two digit tokens; both are
replaced together. A natural-percentage replay must reproduce the
saved first verdict token. The neutral follow-up receives the same
initial-percentage prefix as the contrary follow-up, allowing a
matched-visible-text comparison. The previous system and user messages
stay unchanged; only the supplied assistant answer prefix differs.
Each run records input and script hashes, case-level outputs,
a completion hash, and a summary; interrupted runs
resume only when their manifest matches.

## Reference result

Among the eligible natural accept-to-reject flips, replacing the
written percentage with the initial percentage makes the next verdict
token ACCEPT in:

| Cohort | Admissions | Paper submissions |
| --- | ---: | ---: |
| Steering confirmation | 12.5% (5/40) | 100% (37/37) |
| Independent wording cohort | 23.5% (8/34) | 97.4% (37/38) |

The disjoint cohorts' complete output hashes are
`d2b3bc8bec090c44b20de40e08355870c583ec7e470e62990a74e026bfd5168d`
and `142ceff3a58f5f53112ea15dc0a93a4aaf04b30019a84caef604d9512c183aee`,
respectively. These hashes identify the full research runs, which also
included exploratory 0/50/100 substitutions in the first cohort; the
public runner contains the three conditions needed to reproduce the
paper's comparison. The public analyzer checks the full cohort, both
non-numeric prefix spans, the natural-percentage sham, and the
equal-percentage cases.

At the common forced value of 75%, the next token is ACCEPT in 1/36
admissions and 30/30 paper cases in the confirmation cohort, and 1/27
and 31/32 in the independent cohort. Neutral follow-ups with the same
written initial percentage emit ACCEPT in all 149 eligible cases.
The contrast concerns model reports under these two task settings;
the source cases and prompt wording differ between domains.

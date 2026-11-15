# Reproducing the case-paired wording check

This extension tests whether the block-21 cached value at the final punctuation
token of the model's own answer prefix affects the first numeric token under
four followup wordings. It uses a third, disjoint cohort of 160 cases and the
same Qwen2.5-7B-Instruct revision as the [main route experiment](reproduce-route.md).
The source outcomes are never included in prompts or selected case files.

## Data and environment

Use the source revisions, import commands, and original 300-case sample in the
[main reproduction guide](reproduce-route.md). The imported GradCafe and
PeerRead JSONL files must have SHA-256 digests
`945a7c99f0f04e18f3f243c02bd47651234044aea60c8f3a8ce24141075c8d05`
and `8195b87b0493782ea01195519467b58196cd87825e9e8093c4acf107900fe8c9`.
The original sample digest is
`bd61ddf20c0c54fa80f9be78709a2f7389cfd678bf5c543d9e4ef030c0d15ca0`.
The prior wording cohort's 160 case IDs are in
[`configs/validation-160-selection.json`](../configs/validation-160-selection.json).

Install the package in a Python 3.10+ environment with CUDA and at least 24 GB
of free GPU memory. The reference run used an A40. Set the paths below to your
local checkout and imported files; keep the run directory outside the checkout.

```bash
V2_REPO=/path/to/llm-forecast-revision
V2_WORK=/work/wording-v2
mkdir -p "$V2_WORK"
cp "$V2_REPO/configs/feedback-wording-validation-v1.json" "$V2_WORK/"

python "$V2_REPO/scripts/wording_v2/select_cases.py" \
  --original /work/data/cases-300.jsonl \
  --prior-ids "$V2_REPO/configs/validation-160-selection.json" \
  --gradcafe /work/data/gradcafe.jsonl \
  --peerread /work/data/peerread.jsonl \
  --output "$V2_WORK/cross-wording-v2-cases.jsonl"

cd "$V2_WORK"
python "$V2_REPO/scripts/wording_v2/collect.py" baseline
python "$V2_REPO/scripts/wording_v2/collect.py" variants
python "$V2_REPO/scripts/wording_v2/route.py"
python "$V2_REPO/scripts/wording_v2/analyze.py" \
  --root "$V2_WORK/cross-wording-v2"
```

The selector fixes the case-file SHA-256 at
`8b2dc2783229e172c730e1a4dd8a137de26b4968af4348533ea0813eb80230eb`
and rejects source revisions that would change the cohort. Baseline collection
uses the public `collect` implementation; the exact followup wording is in
[`configs/feedback-wording-validation-v1.json`](../configs/feedback-wording-validation-v1.json).
Each phase records its input and code hashes, so interrupted runs resume only
with the same files. Run one GPU phase at a time.

## Intervention and measurements

The model first reports `P(accept)` and a verdict. A neutral followup asks it
to look again; the other followups are original contrary expectation, two
paraphrases, and an unrelated expectation about answer length. No followup adds
case facts. For every condition with a first number token different from the
neutral token, `route.py` captures the natural generation at that step and
replaces the cached value of the answer prefix's final punctuation token with
the neutral value from the same case. It tests query head 20 alone and all four
KV groups, each against a deterministic equal-norm orthogonal control. The
cached value is restored on the following decoding step. Single-step and
full-answer score and token checks must agree.

The primary measure is the paired-minus-control change in the processed score
margin between that condition's own natural first number token and the neutral
token. Negative values favor the neutral token. The summary also reports raw
logit margins, first-token, full-percentage, and changed-verdict restoration,
domain splits, the common eligible intersection, and 10,000-resample case-bootstrap 95%
intervals. The fixed model revision, block, head, cache position, sample, and
wordings were chosen before this cohort was inferred.

## Reference checks

The completed baseline contains 476 records: 160 first turns and 158 each of
neutral and original followups. Two first-turn admissions responses used an
unparseable `UNDECIDED` verdict; their cases remain in the sample and receive
no second turn. The variant run contains 474 records with no additional parse
failures. The route run contains 160 case records. The completed record hashes
are:

| Phase | SHA-256 of `records.jsonl` |
| --- | --- |
| Baseline | `194543ab49fabd2b8dc68a80ae3c7a7f56dfd401bf13afd3fe91f734279809cf` |
| Variants | `a3433571051e4fe340d1d637fd510bf817a101e5b6fde7afa35dcd586eba6eda` |
| Route | `53bec6571aaf3633583e9d926c8e8b3bbb5ef795fb5f16be44e6ed62c67009b6` |

The [aggregate summary](wording-v2-summary.json) provides exact denominators,
intervals, and domain results. The main comparison is:

| Followup | Eligible | Head-20 net score shift (95% CI) | Full percentage restored: head 20 / control | All values / control |
| --- | ---: | ---: | ---: | ---: |
| Original pushback | 75 | −1.12 [−1.23, −1.03] | 46/75 vs 6/75 | 63/75 vs 4/75 |
| Paraphrase A | 157 | −2.06 [−2.17, −1.95] | 30/157 vs 1/157 | 41/157 vs 0/157 |
| Paraphrase B | 157 | −1.28 [−1.35, −1.22] | 4/157 vs 0/157 | 10/157 vs 0/157 |
| Unrelated challenge | 147 | −1.27 [−1.35, −1.20] | 7/147 vs 0/147 | 17/147 vs 0/147 |

The score effect spans all four wordings, including the unrelated challenge.
Complete-percent restoration depends strongly on wording and domain. The
same 73 cases are eligible under every wording: head 20 restores the full
neutral percentage in 44/73 under original pushback, 1/73 under paraphrase A,
and 0/73 under paraphrase B and the unrelated challenge. The all-group counts
are 61/73, 4/73, 0/73, and 1/73. The earlier fixed-direction experiment and
this case-paired experiment test
different interventions; their results should be read separately.

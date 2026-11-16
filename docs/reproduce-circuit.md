# Reproducing the answer-prefix circuit analysis

This experiment follows a controlled change to the model's answer prefix into
the first number it emits. The visible prefix stays `P(accept): ` throughout.
We replace the block-20 state of its final punctuation token with the state
from the same case's neutral conversation, then restore selected cache entries
or attention-head outputs from the original pushback conversation.

The analysis identifies a local route for the effect of that state replacement.
It measures the first numeric token. The later acceptance/rejection verdict is
studied separately in the [verdict-steering experiment](reproduce-steering.md).

## Samples and environment

Use the package's pinned Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28`, greedy decoding and a CUDA GPU with
at least 24 GiB of free GPU memory. The reference hardware was an A40. Run the GPU
commands sequentially and keep downloaded data and outputs outside the checkout.

The two samples contain 75 cases each. In each selected case, the neutral and
original-pushback answers naturally begin their percentage with different
numeric tokens. The case IDs and source hashes are in
[`configs/circuit/selection.json`](../configs/circuit/selection.json).

| Sample | Admissions | Paper submissions | Baseline to reconstruct |
| --- | ---: | ---: | --- |
| Cache evaluation (`case_paired`) | 37 | 38 | [Case-paired wording baseline](reproduce-wording-v2.md) |
| Wording evaluation (`initial_wording`) | 32 | 43 | Training baseline in the [verdict-steering guide](reproduce-steering.md) |

The samples have no repeated case IDs. Neither selection nor measurement uses
the eventual admission or paper decision. Source records and individual model
answers remain outside the repository.

## Prepare the conversations

First collect the two baselines using the linked guides. Their `records.jsonl`
SHA-256 values must be:

- Case-paired baseline: `194543ab49fabd2b8dc68a80ae3c7a7f56dfd401bf13afd3fe91f734279809cf`.
- Initial-wording baseline: `76c1954d3a44c345ef3556f79adf840a93bd0446cc34160ee61ccd87daa64a05`.

From the repository root:

```bash
CIRCUIT_REPO=/path/to/llm-forecast-revision
CIRCUIT_WORK=/work/answer-prefix-circuit
mkdir -p "$CIRCUIT_WORK"

python "$CIRCUIT_REPO/scripts/circuit_v1/prepare.py" \
  --baseline /work/wording-v2/cross-wording-v2/baseline \
  --selection "$CIRCUIT_REPO/configs/circuit/selection.json" \
  --cohort case_paired \
  --output "$CIRCUIT_WORK/case_paired-cases.jsonl"
python "$CIRCUIT_REPO/scripts/circuit_v1/prepare.py" \
  --baseline /work/verdict-steering/train-baseline \
  --selection "$CIRCUIT_REPO/configs/circuit/selection.json" \
  --cohort initial_wording \
  --output "$CIRCUIT_WORK/initial_wording-cases.jsonl"
```

Preparation checks each baseline's hash and completion marker, every selected
conversation's prompt hash, the task counts and the resulting case-file hash.
The prepared files contain only the conversation fields needed for replay and
the saved numerical reports needed to check the first-token pair.

## Replay the interventions

```bash
for CIRCUIT_COHORT in case_paired initial_wording; do
  python "$CIRCUIT_REPO/scripts/circuit_v1/run.py" \
    --cases "$CIRCUIT_WORK/$CIRCUIT_COHORT-cases.jsonl" \
    --selection "$CIRCUIT_REPO/configs/circuit/selection.json" \
    --cohort "$CIRCUIT_COHORT" \
    --output "$CIRCUIT_WORK/$CIRCUIT_COHORT"
done

python "$CIRCUIT_REPO/scripts/circuit_v1/analyze.py" \
  --case-paired "$CIRCUIT_WORK/case_paired" \
  --initial-wording "$CIRCUIT_WORK/initial_wording" \
  --output "$CIRCUIT_WORK/summary.json"
```

The runner also accepts `--only-case CASE_ID` for checking one selected case in
a separate output directory. The analyzer requires both complete 75-case runs.
Interrupted runs resume with the same input, code and model manifest.

The runner performs three linked measurements for each case:

1. **State replacement and cache restoration.** It captures the natural
   neutral and pushback prefix states, replaces the punctuation state at block
   20, and measures the first number. An equal-norm orthogonal replacement is
   the control. Separate replays restore block-21 group-2 values, its keys and
   values together, later punctuation caches, all those caches together, block
   26 alone, or all later caches except block 26.
2. **Complete block-26 head scan.** With block-21 group-2 keys and values
   restored, it compares the edited run with a block-26 cache restoration.
   Each of the 28 head outputs is replaced separately by its counterpart from
   that restoration. Replacing all 28 must exactly reconstruct the block-26
   cache result.
3. **Head-26 key/value dissection.** It restores the punctuation key, value or
   both at block 26 while allowing head 26 to change and holding the other 27
   head outputs at their edited values. The unchanged-cache replay must match
   the edited baseline; joint key/value restoration must reconstruct the
   previously measured head-26 effect.

All block and head indices are zero-based. The state replacement occurs while
the punctuation token before the final space is processed. Cache and head
restorations occur one forward pass later, immediately before the first numeric
token. At block 21, the other 27 head outputs are held at their natural
pushback values, isolating head 20's contribution at that layer. Later cache
restorations replace the punctuation position in all four key/value groups of
the selected layers. This is a larger intervention than restoring two heads.

The score margin is the processed score of the natural pushback first number
minus that of the neutral first number. It includes the checkpoint's decoding
adjustments. Raw margins are recorded as a second check. A positive restoration
effect moves the score back toward the original pushback run. Visible prefixes,
hook call counts, exact replays and cross-measurement reconstruction are checked
for every case. Hooks are removed after each generation.

## Reference result

The [complete aggregate](circuit-summary.json) reports both task settings,
all 28 heads, key/value conditions and 95% case-bootstrap intervals. The
following values concern the effect of the specified punctuation-state edit:

| Measurement | Cache evaluation (75) | Wording evaluation (75) |
| --- | ---: | ---: |
| State replacement minus orthogonal score shift | −2.783 | −2.778 |
| Exact original raw and processed scores after combined cache restoration | 100% (75/75) | 100% (75/75) |
| Block-26 head-26 output restoration | +1.158 | +1.272 |
| All 28 block-26 head outputs restored | +1.143 | +1.243 |
| Head-26 value-only restoration | +1.168 [1.053, 1.282] | +1.272 [1.175, 1.370] |
| Head-26 key-only restoration | +0.020 [0.001, 0.039] | +0.013 [−0.005, 0.030] |

Head 26 has the largest effect in both head scans. Head 11 is next, at +0.094
and +0.117. The scans were run in both samples before heads were ranked; they
are an exploratory localization across disjoint cases, rather than a held-out
test of a head chosen in advance. The value-only effects point to a carrier of
the local restoration through head 26. Their scale is an intervention effect,
not a measure of how much semantic information a vector contains.

Restoring the combined cache set recovers the original score in every selected
case. The set includes block-21 group-2 keys and values and all punctuation
key/value groups at blocks 22–27. Thus the exact restoration localizes how the
state replacement propagates; it does not assign its full effect to heads 20
and 26 alone. The key/value results and complete scan support a more specific
late-layer contribution through head 26.

The published aggregate was recomputed from the completed state-restoration,
head-scan and key/value GPU runs. Release checks reconstructed both prepared
samples, verified the consolidated intervention functions against those used
in the research runs, and reproduced every reported mean and interval from
the archived case-level measurements. The consolidated public runner has not
received a fresh full CUDA rerun. Its manifest and exact-reconstruction checks
are intended to expose model, library or replay differences during reproduction.

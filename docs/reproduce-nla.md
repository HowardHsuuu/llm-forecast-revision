# Matched-state NLA readouts

This protocol reproduces the report's Natural Language Autoencoder (NLA) comparison: 24 held-out cases, each with a neutral and a contrary-expectation followup. It uses the **third-person initial prompt** and the block-20 state immediately before the first number. The two 12-case phases contain 24 descriptions each. The frozen case IDs and checkpoint revisions are in [matched-selection.json](../configs/nla/matched-selection.json).

## Prepare states

First collect and trace the third-person condition using the source data and pinned target model in [the route guide](reproduce-route.md). From the repository root:

```bash
forecast-mechanisms collect /work/data/cases-300.jsonl /work/runs/third-person --prompt-voice third_person
forecast-mechanisms trace /work/runs/third-person /work/runs/third-person-traces

python scripts/nla_v1/run.py prepare \
  --run /work/runs/third-person \
  --traces /work/runs/third-person-traces \
  --selection configs/nla/matched-selection.json \
  --output /work/runs/nla-matched
```

Preparation checks the source collection and trace manifest against the reference hashes, reconstructs the 48 selected states, and verifies their byte-level identity. Training cases supply the shared-direction diagnostic; the 24 selected cases are held out. The saved input hashes also protect subsequent steps against changes during resume. Float32 projection reductions can vary slightly across CPU libraries, so the locally generated direction array receives its own hash.

This is an exact-reference replay: use the pinned [CUDA reference environment](../configs/reference-environment.json). Other compatible software or GPU configurations can change generated answers or activation bytes; preparation deliberately stops on such a mismatch. Output directory names and timestamps do not enter the reference hashes.

## Describe and reconstruct

Use a Linux CUDA environment. Install the optional NLA dependencies and the pinned [official inference source](https://github.com/kitft/nla-inference/tree/38b802a33d1d317f21b6825a9116f388c2141f86):

```bash
pip install -e '.[nla]'
git clone https://github.com/kitft/nla-inference /work/nla-inference
git -C /work/nla-inference checkout 38b802a33d1d317f21b6825a9116f388c2141f86
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('kitft/nla-qwen2.5-7b-L20-av',
    revision='b88469162777ae6553bc14208eb0cb579336f8f4', local_dir='/work/nla/actor')
snapshot_download('kitft/nla-qwen2.5-7b-L20-ar',
    revision='e2c9e57eac213d37a31612087f645ab6332c1bb6', local_dir='/work/nla/reconstructor')
PY
```

Start the actor server in a separate terminal. Disabling the radix cache is required because each request supplies different activation embeddings.

```bash
python -m sglang.launch_server --model-path /work/nla/actor \
  --port 30000 --disable-radix-cache --mem-fraction-static 0.80 --trust-remote-code
```

Once the server is ready, generate both phases. The actor receives each activation without the case profile, followup text, or condition label. Decoding uses temperature 0, a 200-token maximum, and a fixed shuffled request order.

```bash
for phase in pilot replication; do
  python scripts/nla_v1/run.py decode \
    --output /work/runs/nla-matched --phase "$phase" \
    --actor /work/nla/actor \
    --official-module /work/nla-inference/nla_inference.py
done
```

Stop that actor server with Ctrl-C in its terminal and wait for its GPU memory to be released. The reference A40 run loaded the actor and reconstructor sequentially. Then reconstruct the descriptions:

```bash
for phase in pilot replication; do
  python scripts/nla_v1/run.py reconstruct \
    --output /work/runs/nla-matched --phase "$phase" \
    --reconstructor /work/nla/reconstructor \
    --official-module /work/nla-inference/nla_inference.py
done
python scripts/nla_v1/analyze.py \
  --output /work/runs/nla-matched --run /work/runs/third-person
```

The driver verifies the official inference module and checkpoint metadata. Actor outputs resume by vector ID only when the saved settings match. Missing explanation tags remain explicit parse failures; the summary reports both the total and the successfully reconstructed pairs.

## Measurements

For a case, let `v_neutral` and `v_opposite` denote the measured states and `r_neutral` and `r_opposite` the reconstructions. With each vector individually normalized to unit length, the paired alignment is:

```text
pair_margin = (r_opposite - r_neutral) · (v_opposite - v_neutral)
```

A positive value means that the reconstructed change aligns with the measured neutral-to-pushback change. The reference has positive margins in **11 of 12** pilot pairs and **12 of 12** replication pairs. This statistic differs from projection onto a shared direction averaged over training cases. The latter is included as a separate diagnostic.

[Reference aggregate results](nla-summary.json) include paired alignment, bootstrap intervals, shared and orthogonal projections, and a first-quoted-percentage baseline. Fixed regular expressions detect percentage/format language in all 48 descriptions and explicit feedback markers in none. The exact expressions are included in the summary; this is a lexical check, not human semantic annotation.

The package-level `nla` and `nla-score` commands provide general per-state descriptions and reconstruction scores. The paired statistic above uses this dedicated protocol.

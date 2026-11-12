# Forecast Mechanisms

Research code for studying how an open-weight language model reports a forecast and verdict for an unresolved admission decision, and how its pre-answer representation changes after applicant feedback.

The central target is the model's **own forthcoming report**. A decoded percentage is not a ground-truth probability, and a decodable feature is not automatically a causally used belief. The software records first- and second-round outputs, pre-answer layer states, held-out probes, NLA descriptions, and targeted interventions so these claims can be tested separately.

## Research protocol

1. Prompt Qwen2.5-7B-Instruct with a pending graduate or paper admission case. First-round arms ask for percentage before verdict, verdict before percentage, percentage only, verdict only, or a natural answer.
2. From the exact first response in the percentage-first and verdict-only arms, ask a neutral followup, or tell the model the applicant expected the opposite or same outcome. No new application facts are supplied.
3. Capture the residual stream at the **last prompt token before any answer token**. For the released Qwen NLA, block 20 output is `hidden_states[21]` (index 0 is embedding output).
4. Fit layerwise ridge/logistic probes on training cases to predict the model's later percentage/verdict. Test on held-out case IDs, output-order arms, and second-round turns.
5. Decode selected held-out block-20 states with the official NLA actor, plus same-condition other-case controls. Optionally score actor text with the NLA reconstructor.
6. On held-out prompts, add positive/negative doses along the training-set percentage-probe direction at block 20's final prompt token. Compare with matched orthogonal directions and baseline replays, including percentage-only and verdict-first arms.

The study can support a narrow claim about a readable and causally relevant direction. It cannot identify a complete forecasting algorithm from these measurements alone. NLA descriptions are candidates for interpretation and require controls/fidelity checks.

## Data contract

Input is UTF-8 JSONL, one object per case:

```json
{"case_id":"sample-001","domain":"grad_admissions","context":"An applicant to Example University MS CS has a 3.7 GPA and research experience.","source_ref":"curated:001","outcome":null,"audit_status":"verified_predecision"}
```

`outcome` may be `accept`, `reject`, or `null`; it is stored for later aggregate scoring and is **never** passed into the model prompt. Use `verified_predecision` only after independently checking that the input was available before the decision and is suitable for outcome analysis. Historical GradCafe and PeerRead importers mark rows `legacy_unverified`; their outcome labels are excluded from calibration scoring. Check imported rows for duplicates, implausible fields, and data rights before use. The GradCafe importer excludes comments and decision fields from the prompt; the PeerRead importer uses title and abstract only, excluding reviews and scores.

Keep source datasets and all run outputs outside this repository. The project does not download or redistribute either dataset.

## Setup on one A40 48 GB

Use a Linux CUDA environment with Python 3.10+ and enough disk for the target (~15 GB), NLA actor (~15 GB), optional NLA reconstructor (~11 GB), Python environment, and outputs. A 100 GB volume is a reasonable starting size. Model stages run sequentially: collect/probe/intervene use the target; NLA decoding uses the actor server. Do not keep a target process loaded while starting NLA if memory is tight.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install 'sglang[all]>=0.5.6,<0.6' httpx orjson
```

The package pins the target model to Hugging Face revision `a09a35458c702b33eeacc393d103063234e8bc28`. The [official NLA actor](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-av) and [reconstructor](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-ar) must match Qwen2.5-7B-Instruct, 3584 dimensions, and block 20. Download them outside this repo:

```bash
hf download kitft/nla-qwen2.5-7b-L20-av --revision b88469162777ae6553bc14208eb0cb579336f8f4 --local-dir /work/models/nla-av
hf download kitft/nla-qwen2.5-7b-L20-ar --revision e2c9e57eac213d37a31612087f645ab6332c1bb6 --local-dir /work/models/nla-ar
git clone https://github.com/kitft/nla-inference.git /work/deps/nla-inference
git -C /work/deps/nla-inference rev-parse HEAD
```

Record the printed NLA inference source commit with the study's run notes. The `nla` command uses that repository's `nla_inference.py` directly and reads the actor's `nla_meta.yaml`; it does not reimplement the injection protocol.

## Run

Prepare one or both historical datasets, or supply a separately curated case file. The importers never mark historical rows verified.

```bash
forecast-mechanisms import-gradcafe /work/raw/gradcafe /work/data/gradcafe.jsonl
forecast-mechanisms import-peerread /work/raw/PeerRead/data/iclr_2017 /work/data/peerread.jsonl
forecast-mechanisms sample /work/data/cases.jsonl /work/data/gradcafe.jsonl /work/data/peerread.jsonl --per-domain 150
forecast-mechanisms audit /work/data/cases.jsonl
forecast-mechanisms preflight /work/data/cases.jsonl /work/runs
forecast-mechanisms collect /work/data/cases.jsonl /work/runs/main
forecast-mechanisms probe /work/runs/main /work/runs/probes
forecast-mechanisms analyze /work/runs/main /work/runs/analysis
```

`collect` writes one activation matrix per prompt and resumes safely when called again with the same cases, model, and settings. The case split is deterministic by ID; all arms and turns for one case stay in the same split. A changed case file or setting requires a new run directory. Failed parses remain in the record and are counted, never silently corrected.

After collection, stop the target-model process. Start the NLA actor server using the **same checkpoint path** and disable radix caching, as required by the [official inference instructions](https://github.com/kitft/nla-inference):

```bash
python -m sglang.launch_server --model-path /work/models/nla-av --port 30000 --disable-radix-cache --mem-fraction-static 0.80 --trust-remote-code
```

In another shell:

```bash
forecast-mechanisms nla /work/runs/main /work/runs/nla \
  --actor /work/models/nla-av \
  --official-module /work/deps/nla-inference/nla_inference.py
```

This writes raw/extracted descriptions, matched-vector controls, and separate blind-coding files. After stopping the actor server, score saved explanations in a reconstructor-only pass:

```bash
forecast-mechanisms nla-score /work/runs/main /work/runs/nla /work/runs/nla-fidelity \
  --critic /work/models/nla-ar \
  --official-module /work/deps/nla-inference/nla_inference.py
```

A missing close tag is preserved as a failed parse.

Stop the NLA server before loading the target model again for interventions:

```bash
forecast-mechanisms intervene /work/runs/main /work/runs/probes /work/runs/interventions
forecast-mechanisms analyze /work/runs/main /work/runs/analysis --intervention-dir /work/runs/interventions
```

The intervention stage checks that unmodified replays exactly match collection before applying a direction. Large interventions (over 15% of the source state's norm) are skipped and counted. The output includes all parser failures, abstentions, matched controls, and case-level links needed for analysis.

## Interpretation and reproducibility

- Report first-round and second-round effects separately. A first-round probe failing on neutral second turns means pushback scores from that probe are out of distribution; the code also fits a second-round neutral probe.
- A numerical verdict can be inconsistent with the percentage. Preserve that as data.
- Layerwise decodability does not locate the computation's origin. The block-20 causal test is localized to one residual-stream position; it cannot establish a unique circuit.
- Historical results from GradCafe and PeerRead may be self-reported, sparse, outdated, or known during model training. Use curated predecision snapshots for claims about real-world outcome calibration.
- Save `pip freeze`, GPU model, CUDA version, dataset revision, raw source provenance, and the official NLA inference commit with each competition run.

## Sources

- [Qwen2.5-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Natural Language Autoencoders paper](https://transformer-circuits.pub/2026/nla/index.html) and [inference recipe](https://github.com/kitft/nla-inference)
- [PeerRead](https://github.com/allenai/PeerRead) and [GradCafe admissions data](https://github.com/evansrjames/gradcafe-admissions-data)

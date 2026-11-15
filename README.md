# LLM Forecast Revision

Code for studying how Qwen2.5-7B-Instruct reports a percentage and verdict for pending graduate-admission and paper decisions, and how those reports change after a user states a contrary expectation.

## Finding

In 34 held-out conversations where the model naturally changed its percentage, replacing a middle-layer state reproduces the pushback percentage in 8 cases. Replacing the output of one attention head at the next layer raises full-percentage matches from 8 to 13. The additional effect comes from a value cached in the model's own answer prefix. Replacing that value on the natural pushback trajectory shifts the choice toward the neutral number and restores its first numeric token in 7 cases. On 160 new cases, a fixed value direction learned from the earlier cases restores the neutral full percentage in 22 of 75 eligible conversations under the original pushback wording (20 of 43 paper cases and 2 of 32 admissions cases), versus 1 of 75 orthogonal controls. Other wordings also change reports, but the fixed edit does not restore their full percentages. The result identifies a conversation-dependent route into the reported number.

The study uses 150 GradCafe profiles and 150 PeerRead paper titles and abstracts, with eventual outcomes omitted from prompts. It measures the model's reports at decision time. The [research report](docs/research-report.md) gives the protocol, results, and interpretation. The [reproduction supplement](https://github.com/HowardHsuuu/llm-forecast-revision/releases/download/v0.1.1/forecast-mechanisms-submission-code.zip) includes the exact scripts and aggregate results for the additional 160-case validation.

## Installation

The target model requires a CUDA GPU with approximately 24 GB of available memory; the sprint runs used an A40 48 GB. Use Python 3.10 or newer:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
```

The package pins Qwen2.5-7B-Instruct to revision `a09a35458c702b33eeacc393d103063234e8bc28`. Downloaded models, source datasets, and run outputs belong outside this repository.

## Reproduce the route result

Run these commands from the repository root. Import and sample the two source datasets, then collect first-person conversations and decision-token traces:

```bash
forecast-mechanisms import-gradcafe '/work/raw/gradcafe/Computer Science.csv' /work/data/gradcafe.jsonl
forecast-mechanisms import-peerread /work/raw/PeerRead/data/iclr_2017 /work/data/peerread.jsonl
forecast-mechanisms sample /work/data/cases.jsonl /work/data/gradcafe.jsonl /work/data/peerread.jsonl --per-domain 150
forecast-mechanisms collect /work/data/cases.jsonl /work/runs/first-person --prompt-voice first_person
forecast-mechanisms trace /work/runs/first-person /work/runs/decision-traces
```

Run the three fixed case sets and aggregate their results:

```bash
for phase in pilot replication extension; do
  forecast-mechanisms route /work/runs/first-person /work/runs/decision-traces \
    configs/route-phases.json /work/runs/route/$phase --phase "$phase"
done
forecast-mechanisms route-summary /work/runs/route /work/runs/route/summary.json
```

The commands validate collection and trace hashes, exact baseline replays, token alignment, prompt length, hook state equality, and reconstruction of target attention and final raw logits. Each phase writes a manifest and case-level JSONL. Run the phases sequentially on one GPU. See [reproduction details](docs/reproduce-route.md) for source revisions, expected checks, output fields, and readout commands.

## Other commands

`probe` fits held-out percentage and verdict readouts; `nla` and `nla-score` use the [official Natural Language Autoencoder inference code](https://github.com/kitft/nla-inference); `intervene` runs residual-direction controls. Their roles and command examples are in [reproduction details](docs/reproduce-route.md). The CLI also provides `audit`, `preflight`, and `analyze` for dataset and run checks.

## Data

Input is UTF-8 JSONL, one object per case:

```json
{"case_id":"sample-001","domain":"grad_admissions","context":"An applicant to Example University MS CS has a 3.7 GPA and research experience.","source_ref":"curated:001","outcome":null,"audit_status":"verified_predecision"}
```

The importers omit decision fields, reviews, and comments from model prompts. Historical imports receive `legacy_unverified` status. Use `verified_predecision` only for independently checked prospective snapshots. The project does not redistribute either source dataset.

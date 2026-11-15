# LLM Forecast Revision

Code for studying how Qwen2.5-7B-Instruct reports a percentage and verdict for pending graduate-admission and paper decisions, and how those reports change after a user states a contrary expectation.

## Finding

In 34 held-out conversations where the model naturally changed its percentage, replacing a middle-layer state reproduces the pushback percentage in 8 cases. Replacing the output of one attention head at the next layer raises full-percentage matches from 8 to 13. The additional effect comes from a value cached in the model's own answer prefix. On a first disjoint 160-case cohort, a fixed value direction learned from those earlier pairs restores the neutral full percentage in 22 of 75 eligible conversations under the original pushback wording, versus 1 of 75 orthogonal controls. On a second disjoint 160-case cohort, replacing the cached value with its matched neutral value moves first-number scores toward neutral under all four followup wordings. Under the original wording, head 20 restores the complete neutral percentage in 46 of 75 eligible cases versus 6 of 75 controls; replacing all four cached value groups restores 63 versus 4. Full-percentage restoration is lower under paraphrases and an unrelated answer-length challenge.

A separate direction derived from same-case neutral-minus-pushback residual states controls verdict generation under the original wording. On a third disjoint 160-case cohort, it preserves the initial acceptance verdict in 38 of 82 naturally flipped cases, versus 7 of 82 equal-norm orthogonal controls. A later single-step check finds that edits at the first number and first verdict can each change some complete verdicts, including verdict changes after the percentage is already written. These are intervention effects at output positions; the direction is not established as a natural mediator of pushback.

The study uses GradCafe profiles and PeerRead paper titles and abstracts, with eventual outcomes omitted from prompts. It measures the model's reports at decision time. The [research report](docs/report.pdf) and [text version](docs/research-report.md) give the protocol, results, and interpretation. The [case-paired wording guide](docs/reproduce-wording-v2.md) and [verdict-steering guide](docs/reproduce-steering.md) provide frozen protocols, hashes, aggregate results, and reproduction commands. The [code supplement](https://github.com/HowardHsuuu/llm-forecast-revision/releases/download/v0.1.2/forecast-mechanisms-submission-code.zip) includes the earlier validation scripts and LaTeX source.

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

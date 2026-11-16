# LLM Forecast Revision

Code for studying how Qwen2.5-7B-Instruct reports a percentage and verdict for pending graduate-admission and paper decisions, and how those reports change after a user states a contrary expectation.

## Finding

A model can accommodate a user's expectation in its verdict while leaving its reported acceptance probability unchanged. In five admissions cases, the answer remains at 75% while the verdict changes from acceptance to rejection after pushback that adds no case information; sustained residual steering restores four of those verdicts with the same percentage. Across all 146 initial accepts in that cohort, the direction reduces verdict flips from 56.2% to 30.1%. On fresh cases under the same wording, flips again fall from 52.7% to 22.6%, with one newly induced flip among 23 initial rejects.

The revised answer is already partly predictable before generation. On independent cases, pre-answer states improve percentage prediction over initial-report fields and predict which initially accepting verdicts will survive pushback. NLA reconstructions also retain a component of neutral-to-pushback state changes, while their descriptions emphasize numerical answer construction. The [pre-answer readout guide](docs/reproduce-readouts.md) and [NLA guide](docs/reproduce-nla.md) give the measurements, baselines, and reproduction commands.

The steering effect depends strongly on followup form. In a fixed comparison on 116 fresh parseable cases, an instruction-derived direction reduces the balanced flip rate under two held-out wordings (giving initial acceptance and rejection equal weight) by 3.3 percentage points (95% case-bootstrap interval 0.0–7.1); a textual reminder reduces it by 18.3 points (14.3–22.7). Both interventions continue to follow explicit 10%/90% statistical information and mandatory eligibility constraints. See the [stability reproduction guide](docs/reproduce-stability.md) for the complete comparison and response checks.

Paired interventions identify a value cached from the model's answer prefix that affects the first number. Replacing the block-21 value with its matched neutral counterpart restores the complete neutral percentage in 61.3% of eligible standard-pushback cases, versus 8.0% with equal-norm controls. A further analysis follows a punctuation-state replacement through cached history and identifies an additional attention head at block 26, head 26. Value-only effects repeat across two disjoint 75-case samples; restoring the full tested cache set recovers the unedited digit-score margin in every case. See the [local circuit analysis](docs/reproduce-circuit.md).

Replacing the complete written percentage provides a complementary test of the later verdict: across two disjoint samples it restores acceptance in 98.7% (74/75) of eligible paper cases and 17.6% (13/74) of admissions cases. These results trace influences of dialogue context and the written number on the communicated judgment.

The study uses GradCafe profiles and PeerRead paper titles and abstracts, with eventual outcomes omitted from prompts. Reproduction guides cover the [cache route](docs/reproduce-route.md), [local circuit](docs/reproduce-circuit.md), [case-paired wording check](docs/reproduce-wording-v2.md), [paired-direction steering](docs/reproduce-steering.md), [written-percentage intervention](docs/reproduce-forced-number.md), and [stability comparison](docs/reproduce-stability.md). Each guide specifies the protocol, fixed settings, and reference measurements.

## Installation

The target model requires a CUDA GPU with approximately 24 GiB of free memory; the sprint runs used an A40 48 GB. Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The [reference CUDA environment](configs/reference-environment.json) records the inference software versions. The package pins Qwen2.5-7B-Instruct to revision `a09a35458c702b33eeacc393d103063234e8bc28`. Downloaded models, source datasets, and run outputs belong outside this repository.

## Reproduce the route result

Run these commands from the repository root. Import and sample the two source datasets, then collect first-person conversations and decision-token traces:

```bash
forecast-mechanisms import-gradcafe '/work/raw/gradcafe/Computer Science.csv' /work/data/gradcafe.jsonl
forecast-mechanisms import-peerread /work/raw/PeerRead/data/iclr_2017 /work/data/peerread.jsonl
forecast-mechanisms sample /work/data/cases-300.jsonl /work/data/gradcafe.jsonl /work/data/peerread.jsonl --per-domain 150
forecast-mechanisms collect /work/data/cases-300.jsonl /work/runs/first-person --prompt-voice first_person
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

`probe` fits readouts from prompt states, while `probe-decision` reproduces the report's decision-token readouts. The [matched-state NLA guide](docs/reproduce-nla.md) reproduces the paired descriptions and reconstruction comparison; `nla` and `nla-score` provide general readouts through the [official Natural Language Autoencoder inference code](https://github.com/kitft/nla-inference). `intervene` runs residual-direction controls. Their command examples are in [reproduction details](docs/reproduce-route.md). The CLI also provides `audit`, `preflight`, and `analyze` for dataset and run checks.

## Data

Input is UTF-8 JSONL, one object per case:

```json
{"case_id":"sample-001","domain":"grad_admissions","context":"An applicant to Example University MS CS has a 3.7 GPA and research experience.","source_ref":"curated:001","outcome":null,"audit_status":"verified_predecision"}
```

The importers omit decision fields, reviews, and comments from model prompts. Historical imports receive `legacy_unverified` status. Use `verified_predecision` only for independently checked prospective snapshots. The project does not redistribute either source dataset.

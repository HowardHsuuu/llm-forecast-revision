# Reproducing the feedback route

This protocol reconstructs the reported path from a contrary user expectation to the first numeric token in Qwen2.5-7B-Instruct. It uses the model's own percentage and verdict as measurement targets. The eventual admissions and paper decisions are excluded from every prompt.

## Environment and sources

Use Linux, Python 3.10+, a CUDA GPU with at least 24 GB free memory, and enough disk for the checkpoint, two source datasets, and run outputs. The sprint used an A40 48 GB. Install the project as shown in the [README](../README.md). The target checkpoint is `Qwen/Qwen2.5-7B-Instruct` at revision `a09a35458c702b33eeacc393d103063234e8bc28`.

Source data for the reference run:

| Source | Revision | Imported fields |
| --- | --- | --- |
| [GradCafe admissions compilation](https://github.com/evansrjames/gradcafe-admissions-data) | `629ca5d26b4dae52bc4e5187e1bb7732ab41ccf9` | Computer-science application profile |
| [PeerRead](https://github.com/allenai/PeerRead) | `9bb37751781a900cee9e74ec3105997732c8e8e5` | ICLR 2017 title and abstract |

Import 150 cases per domain with seed 42. The reference selected JSONL has SHA-256 `bd61ddf20c0c54fa80f9be78709a2f7389cfd678bf5c543d9e4ef030c0d15ca0`. Compare this digest before running the model; source-file or importer differences change the case set.

```bash
forecast-mechanisms import-gradcafe '/work/raw/gradcafe/Computer Science.csv' /work/data/gradcafe.jsonl
forecast-mechanisms import-peerread /work/raw/PeerRead/data/iclr_2017 /work/data/peerread.jsonl
forecast-mechanisms sample /work/data/cases.jsonl /work/data/gradcafe.jsonl /work/data/peerread.jsonl --per-domain 150 --seed 42
sha256sum /work/data/cases.jsonl
forecast-mechanisms audit /work/data/cases.jsonl
forecast-mechanisms preflight /work/data/cases.jsonl /work/runs
```

## Collect and intervene

```bash
forecast-mechanisms collect /work/data/cases.jsonl /work/runs/first-person --prompt-voice first_person
forecast-mechanisms trace /work/runs/first-person /work/runs/decision-traces

for phase in pilot replication extension; do
  forecast-mechanisms route /work/runs/first-person /work/runs/decision-traces \
    configs/route-phases.json /work/runs/route/$phase --phase "$phase"
done

forecast-mechanisms route-summary /work/runs/route /work/runs/route/summary.json
```

`configs/route-phases.json` fixes 12 pilot, 12 disjoint replication, and 39 extension case IDs from the common 63-case test split (SHA-256 `691da5562d703f41e5c7291e9e5505dc19451e79520158db2731a90b10957393`). The route command constructs the length-matched neutral followup by inserting a fixed uncertainty sentence. It verifies that neutral and pushback prompts have equal token counts and that the generated prefix before the first number is identical. The pinned checkpoint emits `P(accept): ` as five tokens; the first number is generated at index 5.

For each informative case, the command captures block-20 residual state, block-21 current-token projections and prior-token KV cache, head outputs, and later attention outputs. It performs direct state replacement in both directions; single-head and all-head output patches; past-cache segment patches; one-position value patches; reciprocal patches on natural pushback trajectories; same-domain donor and equal-norm orthogonal vector controls; and cumulative attention-output patches through block 27. Source-state equality, unchanged generated prefix, exact Q/K/V projection agreement after block-20 replacement, all-past attention reconstruction, and final raw-logit reconstruction are executable checks. Results are appended to `records.jsonl`; the manifest records input and intervention-code hashes, so rerunning the same phase resumes only matching work.

## Reference measurements

The reference run has 34 informative cases among 63 held-out pairs. Denominators below refer to those 34 cases unless stated otherwise.

| Measurement | Reference result |
| --- | ---: |
| Block-20 forward replacement reaches pushback percentage | 8/34 |
| Block-20 reverse replacement restores neutral percentage | 32/34 |
| Block-20 equal-norm orthogonal controls, forward / reverse | 1/34 / 6/34 |
| Forward target first-number-token matches: state / head 20 / all block-21 heads | 8/34 / 13/34 / 19/34 |
| Forward target full-percentage matches: state / head 20 | 8/34 / 13/34 |
| Forward target full-percentage matches, admissions / paper: state → head 20 | 7/13 → 7/13 / 1/21 → 6/21 |
| All-past / generated-prefix / user-turn-only block-21 KV patches | 19/34 / 14/34 / 8/34 |
| One `):` prefix-value patch, head 20 isolated | 13/34 |
| Reciprocal natural-pushback `):` patch restores neutral first token | 7/34 |
| Cumulative later attention patches through block 27 | 34/34; target raw logits equal |

The `):` value patch changes the processed target-minus-neutral first-token margin by +0.723 forward and −0.517 on natural pushback trajectories. Same-domain donor differences yield +0.714 and −0.492; equal-norm orthogonal vectors yield −0.085 and +0.021. The donor and recipient differences have mean cosine 0.940. `route-summary` reports case-bootstrap intervals and domain/stage breakdowns alongside these means. Processed margins include the checkpoint's `repetition_penalty=1.05`; raw margins are retained separately.

The route records measure the first numeric token. Full-answer continuation separately verifies the state/head-20 result above. Direct block-20 replacement also generates the full answer so its percentage can be compared with the natural counterpart. The head and cache patches preserve the already-generated answer prefix.

The first 160-case wording validation is reported in [the paper](research-report.md). Its fixed protocol, exact run scripts, aggregate results, and source-data preparation instructions are in the [code supplement](https://github.com/HowardHsuuu/llm-forecast-revision/releases/download/v0.1.2/forecast-mechanisms-submission-code.zip). A further disjoint cohort tests [case-paired cached-value replacements under each wording](reproduce-wording-v2.md).

## Readouts

The collection also supports held-out linear probes:

```bash
forecast-mechanisms probe /work/runs/first-person /work/runs/probes
```

Natural Language Autoencoder descriptions use the [official inference package](https://github.com/kitft/nla-inference). The sprint used actor revision `b88469162777ae6553bc14208eb0cb579336f8f4`, reconstructor revision `e2c9e57eac213d37a31612087f645ab6332c1bb6`, and inference source commit `38b802a33d1d317f21b6825a9116f388c2141f86`. Run the actor and reconstructor sequentially if GPU memory is limited. The package's `nla` and `nla-score` commands wrap that official inference implementation; `--trace-dir ... --position pre_number` targets the same decision step as the route experiment.

Historical source labels have not been independently verified as prospective outcomes. The study's reported percentages and internal interventions are therefore analyzed as model behavior, with no outcome-based accuracy score.

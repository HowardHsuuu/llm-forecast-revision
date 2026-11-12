"""Command line entry points for the forecast-mechanisms workflow."""

from __future__ import annotations

import argparse
import json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecast-mechanisms")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("import-gradcafe", "import-peerread"):
        p = sub.add_parser(name)
        p.add_argument("source")
        p.add_argument("output")
        p.add_argument("--limit", type=int)
    p = sub.add_parser("sample")
    p.add_argument("output")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--per-domain", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p = sub.add_parser("audit")
    p.add_argument("cases")
    p = sub.add_parser("preflight")
    p.add_argument("cases")
    p.add_argument("work_dir")
    p = sub.add_parser("collect")
    p.add_argument("cases")
    p.add_argument("output_dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p = sub.add_parser("probe")
    p.add_argument("run_dir")
    p.add_argument("output_dir")
    p = sub.add_parser("nla")
    p.add_argument("run_dir")
    p.add_argument("output_dir")
    p.add_argument("--actor", required=True)
    p.add_argument("--official-module", required=True)
    p.add_argument("--server-url", default="http://localhost:30000")
    p.add_argument("--critic")
    p.add_argument("--max-per-group", type=int, default=20)
    p = sub.add_parser("nla-score")
    p.add_argument("run_dir")
    p.add_argument("nla_dir")
    p.add_argument("output_dir")
    p.add_argument("--critic", required=True)
    p.add_argument("--official-module", required=True)
    p = sub.add_parser("intervene")
    p.add_argument("run_dir")
    p.add_argument("probe_dir")
    p.add_argument("output_dir")
    p = sub.add_parser("analyze")
    p.add_argument("run_dir")
    p.add_argument("output_dir")
    p.add_argument("--intervention-dir")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command in ("import-gradcafe", "import-peerread"):
        from .datasets import import_gradcafe, import_peerread
        function = import_gradcafe if args.command == "import-gradcafe" else import_peerread
        result = function(args.source, args.output, args.limit)
    elif args.command == "sample":
        from .datasets import sample_cases
        result = sample_cases(args.inputs, args.output, args.per_domain, args.seed)
    elif args.command == "audit":
        from collections import Counter
        from .core import read_jsonl, validate_cases
        cases = read_jsonl(args.cases)
        validate_cases(cases)
        result = {
            "cases": len(cases),
            "domains": dict(Counter(r["domain"] for r in cases)),
            "outcomes": dict(Counter(r.get("outcome") for r in cases)),
            "audit_status": dict(Counter(r["audit_status"] for r in cases)),
        }
    elif args.command == "preflight":
        from .preflight import preflight
        result = preflight(args.cases, args.work_dir)
    elif args.command == "collect":
        from .collect import collect
        from .model import ForecastRunner
        runner = ForecastRunner(max_new_tokens=args.max_new_tokens)
        result = collect(args.cases, args.output_dir, runner, seed=args.seed)
    elif args.command == "probe":
        from .probes import fit_probes
        result = fit_probes(args.run_dir, args.output_dir)
        result = {"fits": {key: {k: v for k, v in value.items() if k != "layer_curve"} for key, value in result["fits"].items()}}
    elif args.command == "nla":
        from .nla_readout import decode_nla
        result = decode_nla(
            args.run_dir, args.output_dir, actor_checkpoint=args.actor,
            official_module=args.official_module, server_url=args.server_url,
            critic_checkpoint=args.critic, max_per_group=args.max_per_group,
        )
    elif args.command == "nla-score":
        from .nla_readout import score_saved_nla
        result = score_saved_nla(
            args.run_dir, args.nla_dir, args.output_dir,
            critic_checkpoint=args.critic, official_module=args.official_module,
        )
    elif args.command == "intervene":
        from .intervene import intervene
        from .model import ForecastRunner
        runner = ForecastRunner()
        result = intervene(args.run_dir, args.probe_dir, args.output_dir, runner)
    else:
        from .analyze import write_analysis
        result = write_analysis(args.run_dir, args.output_dir, args.intervention_dir)
        result = {"analysis_file": f"{args.output_dir}/analysis.json", "n_records": result["behavior"]["n_records"]}
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

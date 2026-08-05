"""Aggregate eval result JSONs into a side-by-side comparison table.

Reads the `*_results.json` files written by eval_LidarSingleStep.py under a
results root, keeps only the models you ask for, and prints a pandas table
(optionally exporting CSV).

Examples:
    python -m evaluate.compare_models
    python -m evaluate.compare_models --base-dir results/eval \\
        --models "oursGraph/.../hw0_26OurGraphModel=Ours" \\
                 "GA3CPolicy/GA3C_CADRL/hw0_26GA3C-CARL=GA3C" --csv out.csv
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

# Default results root and the model subpath -> display-label map. Override both
# from the CLI (--base-dir / --models / --models-json).
DEFAULT_BASE_DIR = "evaluate/results/eval"
DEFAULT_MODELS = {
    "2024-07-24_baseline_07h-03m-34s/hw0_26baseline_net": "NAV-DRL",
    "GA3CPolicy/GA3C_CADRL/hw0_26GA3C-CARL": "GA3CPolicy",
}

SCENARIO_TYPES = ["random", "circle", "doorway", "hallway"]


def extract_data(json_file):
    """Pull the scalar eval metrics out of one *_results.json file."""
    with json_file.open("r") as f:
        data = json.load(f)

    # Usually there's only one key in the data, which corresponds to the model's path
    model_key = next(iter(data.keys()))
    eval_data = data[model_key]["0"]
    scenario = next(iter(eval_data.keys()))
    eval_data = eval_data[scenario]
    results = {
        "eval_reward": eval_data["eval reward"][0],
        "eval_collisions": eval_data["eval collisions"][0],
        "eval_collisions_rate": eval_data["eval collisions rate"][0] * 100,
        "eval_stuck_rate": eval_data["eval stuck rate"][0] * 100,
        "eval_success_rate": eval_data["eval success rate"][0] * 100,
        "eval_extra_time": eval_data["eval extra time"][0],
        "eval_vel_mean": eval_data["eval vel mean"][0],
    }
    if "eval step_count" in eval_data:
        results["eval_step_count"] = eval_data["eval step_count"][0]
    else:
        results["eval_step_count"] = None
    return results


def collect_results(base_dir, models_to_test):
    """Walk `base_dir` for result JSONs belonging to the requested models."""
    results = []
    for scenario_path in Path(base_dir).rglob("*_results.json"):
        if "combined_" in scenario_path.name:
            continue
        scenario_name = scenario_path.parent.name
        model_name = None
        for model in models_to_test:
            if re.search(rf"(^|/){re.escape(model)}(/|$)", scenario_path.as_posix()):
                model_name = model
                break
        if model_name is None:
            continue
        data = extract_data(scenario_path)
        data["scenario"] = scenario_name
        data["model"] = models_to_test[model_name]
        results.append(data)
    return results


def compare(base_dir, models_to_test, csv_path=None):
    results = collect_results(base_dir, models_to_test)
    df = pd.DataFrame(results)
    if df.empty:
        print(
            f"No matching results under '{base_dir}' for models: "
            f"{list(models_to_test)}"
        )
        return df

    df = df.sort_values(by=["scenario", "model"])
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.expand_frame_repr", False)

    print(df)
    print("\n" + "=" * 50 + "\n")
    for scenario_type in SCENARIO_TYPES:
        scenario_df = df[df["scenario"].str.contains(scenario_type)]
        if scenario_df.empty:
            continue
        print(f"Results for scenario type: {scenario_type}")
        print(scenario_df)
        print("\n" + "=" * 50 + "\n")

    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"Wrote combined comparison table to {csv_path}")
    return df


def _parse_models(items):
    """Parse ["subpath=Label", ...] (label defaults to the subpath)."""
    out = {}
    for item in items:
        key, _, label = item.partition("=")
        out[key] = label or key
    return out


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate eval result JSONs into a comparison table."
    )
    parser.add_argument(
        "--base-dir", default=DEFAULT_BASE_DIR,
        help="Root directory to search for *_results.json "
             "(default: evaluate/results/eval).",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="PATH=LABEL",
        help="Model subpath=DisplayLabel entries to include "
             "(default: the built-in set).",
    )
    parser.add_argument(
        "--models-json", default=None,
        help="JSON file mapping model subpath -> display label "
             "(alternative to --models).",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Optional path to write the combined table as CSV.",
    )
    return parser.parse_args()


def main(args=None):
    base_dir = DEFAULT_BASE_DIR
    models = dict(DEFAULT_MODELS)
    csv_path = None
    if args is not None:
        base_dir = args.base_dir
        if args.models_json:
            with open(args.models_json) as f:
                models = json.load(f)
        elif args.models:
            models = _parse_models(args.models)
        csv_path = args.csv
    compare(base_dir, models, csv_path)


if __name__ == "__main__":
    main(_parse_args())

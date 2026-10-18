import argparse
import csv
import json
import sys
from pathlib import Path

MASS_DIR = Path(__file__).resolve().parent
MODULE_DIR = MASS_DIR / "Path_Generation_Module"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from Entity_Selection import rank_paths
from Random_Sampling import sample_paths
from Stealthiness import annotate_stealthiness, read_training_triples


def load_targets(path: Path) -> list[tuple[str, str, str]]:
    
    """READ TAB-SEPARATED TARGET TRIPLES"""
    results = []
    seen = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            if len(row) != 3 or not all(column.strip() for column in row):
                raise ValueError(f"{path}:{line_number}: EXPECTED 3 TAB-SEPARATED FIELDS")
            target = tuple(column.strip() for column in row)
            if target not in seen:
                seen.add(target)
                results.append(target)
    if not results:
        raise ValueError(f"NO TARGET TRIPLES FOUND IN {path}")
    return results


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    train_path = MASS_DIR / "dataset_uml" / "train.txt"
    checkpoint = args.checkpoint or MASS_DIR / "outputs" / "Path_Generation" / "transe_umls.pt"
    if not train_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("UMLS TRAIN.TXT OR TRAINED TRANSE CHECKPOINT IS MISSING")

    targets = load_targets(args.targets)
    training_triples = read_training_triples(train_path)
    training_set = set(training_triples)
    report = []

    for index, (head, relation, tail) in enumerate(targets, 1):
        output_dir = args.output_dir / f"target_{index:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{index}/{len(targets)}] TARGET: ({head}, {relation}, {tail})", flush=True)
        try:
            entity_output = rank_paths(
                head, relation, tail,
                threshold=args.threshold,
                top_k=args.top_k,
                top_entities=args.top_entities,
                checkpoint=checkpoint,
            )
            save_json(output_dir / "entity_selection_output.json", entity_output)

            # USE THE EXISTING PAPER-INSPIRED RANDOM SAMPLING FUNCTION.
            sampling_output = sample_paths(
                entity_output, delta=args.delta,
                max_paths=args.max_paths, seed=args.seed,
            )
            save_json(output_dir / "random_sampling_output.json", sampling_output)

            # STEALTHINESS MODULE ONLY ANNOTATES ENTITY DEGREES.
            stealth_output = annotate_stealthiness(sampling_output, training_triples)
            save_json(output_dir / "stealthiness_output.json", stealth_output)

            # DIAGNOSTIC ONLY: REPORT DISTINCT NEW TRIPLES, DO NOT INJECT.
            selected_paths = sampling_output["selected_candidate_paths"]
            unique_new = set()
            for candidate in selected_paths:
                for key in ("first_triple", "second_triple"):
                    triple = tuple(candidate[key])
                    if triple not in training_set:
                        unique_new.add(triple)
            (output_dir / "new_triples_diagnostic.tsv").write_text(
                "".join("\t".join(triple) + "\n" for triple in sorted(unique_new)),
                encoding="utf-8",
            )
            record = {
                "target": [head, relation, tail],
                "status": "ok",
                "ranked_paths": entity_output["num_ranked_paths"],
                "selected_paths": sampling_output["num_selected_paths"],
                "distinct_new_triples": len(unique_new),
                "output_dir": str(output_dir),
            }
            print(f"  SELECTED PATHS: {record['selected_paths']} | DISTINCT NEW TRIPLES: {len(unique_new)}")
        except (ValueError, KeyError, FileNotFoundError) as exc:
            record = {"target": [head, relation, tail], "status": "error", "error": str(exc)}
            print(f"  ERROR: {exc}")
        report.append(record)

    save_json(args.output_dir / "batch_summary.json", {
        "targets_file": str(args.targets),
        "total_targets": len(targets),
        "successes": sum(item["status"] == "ok" for item in report),
        "results": report,
    })
    print(f"\nBATCH SUMMARY: {args.output_dir / 'batch_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="BATCH UMLS MASS PATH GENERATION")
    parser.add_argument("--targets", type=Path, default=MASS_DIR / "target_triples.txt")
    parser.add_argument("--output-dir", type=Path, default=MASS_DIR / "outputs" / "Batch_Targets")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="CURRENT PROJECT DEFAULT; PAPER USES CONFIDENCE > 0.9")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-entities", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.7)
    parser.add_argument("--max-paths", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
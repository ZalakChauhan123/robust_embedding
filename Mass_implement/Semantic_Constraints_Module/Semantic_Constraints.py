import os
import sys
from collections import defaultdict


# ============================================================
# PATH CONFIGURATION
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Semantic_Constraints_Module -> Mass_implement
MASS_DIR = os.path.dirname(CURRENT_DIR)

# Mass_implement/dataset_uml
DATASET_DIR = os.path.join(MASS_DIR, "dataset_uml")

# Allow Python to import PRA package
if MASS_DIR not in sys.path:
    sys.path.append(MASS_DIR)


# ============================================================
# IMPORT PRA
# ============================================================

from PRA_Module.Pra_implement_finalVersion import (
    load_dataset,
    MaSSPathTranslator
)


# ============================================================
# 1. NORMALIZE UMLS SEMANTIC TYPES
# ============================================================

def normalize_semantic_type(name):
    """
    Convert UMLS Semantic Network names into the format used
    by our UMLS benchmark.

    Examples:

        Disease or Syndrome
        -> disease_or_syndrome

        Amino Acid, Peptide, or Protein
        -> amino_acid_peptide_or_protein
    """

    return (
        name.strip()
        .lower()
        .replace(",", "")
        .replace(" ", "_")
    )


# ============================================================
# 2. LOAD SRSTRE2
# ============================================================

def load_semantic_constraints(srstre2_path):
    """
    Load domain and range restrictions from SRSTRE2.

    SRSTRE2 format:

        SourceType|Relation|TargetType|

    Example:

        Antibiotic|prevents|Disease or Syndrome|

    Result:

        constraints[relation]["domain"]
        constraints[relation]["range"]
    """

    constraints = defaultdict(
        lambda: {
            "domain": set(),
            "range": set()
        }
    )

    with open(srstre2_path, "r", encoding="utf-8") as file:

        for line in file:
            line = line.strip()

            if not line:
                continue
            parts = line.split("|")

            if len(parts) < 3:
                continue

            source_type = normalize_semantic_type(
                parts[0]
            )

            relation = parts[1].strip().lower()

            target_type = normalize_semantic_type(
                parts[2]
            )

            # Allowed source types
            constraints[relation]["domain"].add(
                source_type
            )

            # Allowed tail types
            constraints[relation]["range"].add(
                target_type
            )

    return constraints


# ============================================================
# 3. LOAD DATASET ENTITIES
# ============================================================

def load_dataset_entities(entities_path):

    entities = set()
    with open(entities_path, "r", encoding="utf-8") as file:
        for line in file:
            entity = line.strip()
            if entity:
                entities.add(entity)
    return entities


# ============================================================
# 4. MASS SEMANTIC CONSTRAINT
# ============================================================

def find_valid_intermediate_entities(
    r1,
    r2,
    constraints,
    dataset_entities
):
    """
    MaSS Semantic Constraint:

        e_middle ∈ Range(r1) ∩ Domain(r2)

    """

    if r1 not in constraints:
        print(
            f"Warning: relation '{r1}' "
            f"not found in SRSTRE2."
        )
        return set()

    if r2 not in constraints:
        print(
            f"Warning: relation '{r2}' "
            f"not found in SRSTRE2."
        )
        return set()

    # Intermediate entity is tail of r1
    range_r1 = constraints[r1]["range"]

    # Intermediate entity is source of r2
    domain_r2 = constraints[r2]["domain"]

    # MaSS semantic constraint
    valid_intermediate_types = (
        range_r1.intersection(domain_r2)
    )

    # Keep only entities existing in our benchmark
    valid_intermediate_entities = (
        valid_intermediate_types.intersection(
            dataset_entities
        )
    )

    return valid_intermediate_entities


# ============================================================
# 5. CONNECT PRA -> SEMANTIC CONSTRAINTS
# ============================================================

def apply_semantic_constraints_to_pra(
    pra_paths,
    constraints,
    dataset_entities
):
    """
    Takes output directly from:

        MaSSPathTranslator.translate_relation()

    PRA output format:

        [
            (
                (r1, r2),
                confidence,
                matching_pairs
            ),
            ...
        ]

    For every PRA relation template, apply:

        Range(r1) ∩ Domain(r2)

    and return semantically valid intermediate entities.
    """

    semantic_results = []

    for path_template, confidence, matching_pairs in pra_paths:

        r1, r2 = path_template

        valid_entities = (
            find_valid_intermediate_entities(
                r1,
                r2,
                constraints,
                dataset_entities
            )
        )

        semantic_results.append(
            {
                "r1": r1,
                "r2": r2,
                "confidence": confidence,
                "valid_intermediate_entities":
                    sorted(valid_entities),
                "num_valid_entities":
                    len(valid_entities)
            }
        )

    return semantic_results


# ============================================================43zzhuhuzfgh
# 6. SAVE RESULTS
# ============================================================

def save_semantic_results(
    results,
    output_path,
    target_relation
):

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "========================================\n"
        )

        file.write(
            "MaSS Semantic Constraints Output\n"
        )

        file.write(
            "========================================\n\n"
        )

        file.write(
            f"Target Relation: {target_relation}\n"
        )

        file.write(
            f"PRA Templates: {len(results)}\n\n"
        )

        for index, result in enumerate(
            results,
            start=1
        ):

            file.write(
                "-" * 70 + "\n"
            )

            file.write(
                f"Template {index}\n"
            )

            file.write(
                f"Path: "
                f"{result['r1']} -> "
                f"{result['r2']}\n"
            )

            file.write(
                f"PRA Confidence: "
                f"{result['confidence']:.4f}\n"
            )

            file.write(
                f"Valid Intermediate Entities: "
                f"{result['num_valid_entities']}\n"
            )

            for entity in result[
                "valid_intermediate_entities"
            ]:

                file.write(
                    f"    {entity}\n"
                )

            file.write("\n")


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # FILE PATHS
    # ========================================================

    TRAIN_PATH = os.path.join(
        DATASET_DIR,
        "train.txt"
    )

    ENTITIES_PATH = os.path.join(
        DATASET_DIR,
        "entities.txt"
    )

    SRSTRE2_PATH = os.path.join(
        DATASET_DIR,
        "Semantic_Constraints",
        "SRSTRE2"
    )

    OUTPUT_DIR = os.path.join(
        MASS_DIR,
        "outputs"
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    SEMANTIC_OUTPUT_PATH = os.path.join(
        OUTPUT_DIR,
        "semantic_constraints_output.txt"
    )


    # ========================================================
    # TARGET RELATION
    # ========================================================

    TARGET_RELATION = "interacts_with"
    PRA_THRESHOLD = 0.8


    # ========================================================
    # STEP 1: LOAD KNOWLEDGE GRAPH
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "STEP 1 - Load UMLS Knowledge Graph"
    )

    print(
        "======================================"
    )

    clean_kg = load_dataset(
        TRAIN_PATH
    )


    # ========================================================
    # STEP 2: RUN PRA
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "STEP 2 - Run PRA"
    )

    print(
        "======================================"
    )

    translator = MaSSPathTranslator(
        clean_kg,
        threshold=PRA_THRESHOLD
    )

    pra_paths = translator.translate_relation(
        TARGET_RELATION
    )

    print(
        f"\nReliable PRA templates found: "
        f"{len(pra_paths)}"
    )


    # ========================================================
    # STEP 3: LOAD SRSTRE2
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "STEP 3 - Load Semantic Constraints"
    )

    print(
        "======================================"
    )

    constraints = load_semantic_constraints(
        SRSTRE2_PATH
    )

    print(
        f"Loaded constraints for "
        f"{len(constraints)} relations."
    )


    # ========================================================
    # STEP 4: LOAD BENCHMARK ENTITIES
    # ========================================================

    dataset_entities = load_dataset_entities(
        ENTITIES_PATH
    )

    print(
        f"Loaded {len(dataset_entities)} "
        f"UMLS benchmark entities."
    )


    # ========================================================
    # STEP 5: PRA -> SEMANTIC CONSTRAINTS
    # ========================================================

    print(
        "\n======================================"
    )

    print(
        "STEP 4 - Apply Semantic Constraints"
    )

    print(
        "======================================"
    )

    semantic_results = (
        apply_semantic_constraints_to_pra(
            pra_paths,
            constraints,
            dataset_entities
        )
    )


    # ========================================================
    # STEP 6: DISPLAY RESULTS
    # ========================================================

    for index, result in enumerate(
        semantic_results,
        start=1
    ):

        print(
            "\n--------------------------------------"
        )

        print(
            f"PRA Template {index}: "
            f"{result['r1']} -> "
            f"{result['r2']}"
        )

        print(
            f"PRA Confidence: "
            f"{result['confidence']:.4f}"
        )

        print(
            "\nValid Intermediate Entities:"
        )

        if result["valid_intermediate_entities"]:

            for entity in result[
                "valid_intermediate_entities"
            ]:

                print(
                    f"  {entity}"
                )

        else:

            print(
                "  No semantically valid "
                "intermediate entity."
            )

        print(
            f"\nTotal: "
            f"{result['num_valid_entities']}"
        )


    # ========================================================
    # STEP 7: SAVE RESULTS
    # ========================================================

    save_semantic_results(
        semantic_results,
        SEMANTIC_OUTPUT_PATH,
        TARGET_RELATION
    )

    print(
        "\n======================================"
    )

    print(
        "Pipeline completed."
    )

    print(
        f"Results saved to:\n"
        f"{SEMANTIC_OUTPUT_PATH}"
    )

    print(
        "======================================\n"
    )
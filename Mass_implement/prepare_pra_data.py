import os


def extract_unique_elements(input_files):

    """
    Read triplet files and parse out all unique entities and relations.
    Returns = 2 sorted lists: (sorted_entities, sorted_relations)
    """
    entities = set()
    relations = set()

    for file_name in input_files:
        if os.path.exists(file_name):
            print(f"  [info] Parsing {file_name} ...")
            with open(file_name, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) == 3:
                        head, relation, tail = parts
                        entities.add(head)
                        relations.add(relation)
                        entities.add(tail)
        else:
            print(f"  [warn] Dataset file '{file_name}' not found, skipping.")

    # Sorting alphabetically keeps dictionary assignments stable and deterministic
    return sorted(list(entities)), sorted(list(relations))


def save_dictionary_file(output_path, items):
    """Write elements sequentially to a text file, one per line."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(f"{item}\n")
        print(f"  [success] Saved index directory to: {output_path}")
    except Exception as exc:
        print(f"  [error] Failed writing dictionary to {output_path}: {exc}")
        raise

def main():

    # 1. Define configuration paths 
    dataset_files = ['dataset_uml/train.txt', 'dataset_uml/valid.txt', 'dataset_uml/test.txt']
    entities_output = 'dataset_uml/entities.txt'
    relations_output = 'dataset_uml/relations.txt'

    print("[step1] Starting extraction of graph vocabulary mappings...")
    
    # 2. Extract elements
    unique_entities, unique_relations = extract_unique_elements(dataset_files)
    
    print(f"[step1] Processing completed.")
    print(f"        Total unique entities discovered: ( {len(unique_entities)} )")
    print(f"        Total unique relations discovered: ( {len(unique_relations)} )")

    # 3. Serialize arrays into distinct text rows
    if unique_entities:
        save_dictionary_file(entities_output, unique_entities)
    else:
        print("[step1] [warn] No entities extracted. Check your file pathways.")

    if unique_relations:
        save_dictionary_file(relations_output, unique_relations)
    else:
        print("[step1] [warn] No relations extracted. Check your file pathways.")


if __name__ == "__main__":
    main()



import collections

"""
PURPOSE - Convert Entities & Relations From "String" -----> Numerical IDs
------------------
load_index_map: Reads the text strings from your entities.txt and relations.txt files and maps them to unique numerical IDs.
------------------
build_relation_adjacency_list: Parses the triplet dataset (train.txt) and constructs an adjacency list representing the graph in memory.
------------------
main: The coordinator that wires these steps together.
------------------
"""


def load_index_map(file_path):
    # Read a vocabulary file (entities or relations) and map each string to a unique numerical ID (0, 1, 2...).
    # Returns a dictionary mapping string -> ID.

    mapping = {}
    print(f"  [info] Building index map from {file_path} ...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                item = line.strip()
                if item:
                    mapping[item] = index
        return mapping
    except Exception as exc:
        print(f"  [error] Failed to load {file_path}: {exc}")
        raise

def build_relation_adjacency_list(triplets_file, entity_map, relation_map):

    # Parse the clean training triplets and build a relation-constrained adjacency list.
    # The structure maps: 
    # source_entity_id -> relation_id -> list of target_entity_ids

    adj_list = collections.defaultdict(lambda: collections.defaultdict(list))
    
    print(f"  [info] Building relation-aware adjacency list from {triplets_file} ...")
    try:
        with open(triplets_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 3:
                    head, relation, tail = parts
                    
                    # Convert string tokens to their assigned numerical IDs
                    head_id = entity_map.get(head)
                    rel_id = relation_map.get(relation)
                    tail_id = entity_map.get(tail)
                    
                    # Ensure all tokens exist in our predefined dictionary mapping
                    if head_id is not None and rel_id is not None and tail_id is not None:
                        adj_list[head_id][rel_id].append(tail_id)
                        
        return adj_list
    except Exception as exc:
        print(f"  [error] Failed building adjacency list: {exc}")
        raise

def print_converted_sample(train_path, entity_map, relation_map, sample_size=5):
    """Print a small sample of the dataset showing how strings map to IDs."""
    print("\n--- Sample of Converted Triplets ---")
    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) == 3:
                    head, relation, tail = parts
                    
                    # Get the IDs mapped in Step 2
                    head_id = entity_map.get(head)
                    rel_id = relation_map.get(relation)
                    tail_id = entity_map.get(tail)
                    
                    print(f"Original Text: ({head}, {relation}, {tail})")
                    print(f"Converted IDs: ({head_id}, {rel_id}, {tail_id})")
                    print("-" * 50)
                    
                    count += 1
                    if count >= sample_size:
                        break
    except Exception as exc:
        print(f"  [error] Failed reading triplets sample: {exc}")

def main():

    # 1. Define configuration paths matching your 'data/' folder
    entities_path = 'dataset_uml/entities.txt'
    relations_path = 'dataset_uml/relations.txt'
    train_path = 'dataset_uml/train.txt'

    print("---- [step2] ---- Initializing the Graph Structure in memory...")

    # 2. Assign unique numerical IDs to your entities and relations
    entity_to_id = load_index_map(entities_path)
    relation_to_id = load_index_map(relations_path)
    
    print(f"Loaded {len(entity_to_id)} entities and {len(relation_to_id)} relations into memory mappings.")

    # 3. Construct the relation-aware adjacency graph
    graph = build_relation_adjacency_list(train_path, entity_to_id, relation_to_id)
    
    # 4. Print validation summary of the initialized graph
    print(f"---- [step2] ---- Graph structure successfully initialized in memory.")
    print(f"Active source entities in adjacency list: {len(graph)}")
    
    # Optional debug print: check the neighbors of the first entity in the map
    first_entity_str = list(entity_to_id.keys())[0]
    first_entity_id = entity_to_id[first_entity_str]
    if first_entity_id in graph:
        sample_relations = list(graph[first_entity_id].keys())
        print(f"[Sample check] Entity '{first_entity_str}' (ID: {first_entity_id}) "
              f"has active outgoing paths via {len(sample_relations)} different relation types.")
    
    # Print sample IDs of Entities & Relations
    print_converted_sample(train_path, entity_to_id, relation_to_id, sample_size=5)


if __name__ == "__main__":
    main()
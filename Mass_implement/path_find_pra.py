import graph_initializer


def discover_paths_dfs(graph, current_node, target_node, max_depth, current_path=None, visited_nodes=None):

    # Recursively search for all unique relation paths up to max_depth
    # connecting a source node to a target node using Depth-First Search (DFS).
    # Returns a set of tuple paths, where each tuple is a sequence of relation IDs.
    

    if current_path is None:
        current_path = []
    if visited_nodes is None:
        visited_nodes = set()

    # If we reached the target node, we found a valid path sequence!
    if current_node == target_node and len(current_path) > 0:
        return {tuple(current_path)}

    # Stop searching if we hit our depth limit
    if len(current_path) >= max_depth:
        return set()

    discovered_paths = set()
    visited_nodes.add(current_node)

    # Explore outgoing edges from the current node
    if current_node in graph:
        for relation_id, neighbor_list in graph[current_node].items():
            for neighbor in neighbor_list:
                # Prevent infinite loops in cyclic graphs
                if neighbor not in visited_nodes or neighbor == target_node:
                    # Recursive call moving to the neighbor node
                    new_paths = discover_paths_dfs(
                        graph=graph,
                        current_node=neighbor,
                        target_node=target_node,
                        max_depth=max_depth,
                        current_path=current_path + [relation_id],
                        visited_nodes=visited_nodes.copy()
                    )
                    discovered_paths.update(new_paths)

    return discovered_paths


def translate_path_to_strings(paths, relation_id_to_str):

    # Convert a set of numeric relation ID paths back into human-readable text.
    readable_paths = []
    for path in paths:
        path_str_list = [relation_id_to_str[rel_id] for rel_id in path]
        readable_paths.append(" -> ".join(path_str_list))
    return sorted(readable_paths)


def main():

    # 1. Reuse Step 2 initialization code to load the graph
    entities_path = 'dataset_uml/entities.txt'
    relations_path = 'dataset_uml/relations.txt'
    train_path = 'dataset_uml/train.txt'

    entity_to_id = graph_initializer.load_index_map(entities_path)
    relation_to_id = graph_initializer.load_index_map(relations_path)
    graph = graph_initializer.build_relation_adjacency_list(train_path, entity_to_id, relation_to_id)
    
    # Reverse relation mapping to easily translate IDs back to text strings
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    print("\n[step3] Starting Path Discovery (Bounded DFS)...")

    # 2. Pick a source and target entity pair to test
    # Based on your dataset output, let's trace paths from 'acquired_abnormality' to a target
    source_entity = 'acquired_abnormality'
    target_entity = 'experimental_model_of_disease'
    
    source_id = entity_to_id.get(source_entity)
    target_id = entity_to_id.get(target_entity)

    if source_id is None or target_id is None:
        print(f"  [error] Source '{source_entity}' or target '{target_entity}' not found in entities.txt")
        return

    # 3. Discover all paths up to path length l = 3
    max_path_length = 3
    print(f"  [info] Searching paths of length 1 to {max_path_length} between:")
    
    unique_paths = discover_paths_dfs(graph, source_id, target_id, max_depth=max_path_length)

    # 4. Print Results
    print(f"\n[step3] Path Discovery Completed!")
    print(f"        Found {len(unique_paths)} unique path types (rules) linking these entities:")
    
    readable_results = translate_path_to_strings(unique_paths, id_to_relation)
    for index, path_str in enumerate(readable_results, 1):
        print(f"        Path Type #{index}: {path_str}")


if __name__ == "__main__":
    main()
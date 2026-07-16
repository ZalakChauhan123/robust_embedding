import graph_initializer


def discover_paths_dfs_with_nodes(graph, current_node, target_node, max_depth, current_path=None, visited_nodes=None):
    """
    Recursively searches for all paths up to max_depth.
    Keeps track of both the relations AND the intermediate nodes.
    
    Returns:
        A list of tuples: [ (entity_0, relation_1, entity_1, relation_2, ..., target_node), ... ]
    """
    if current_path is None:
        # We start by putting the first node in the path list
        current_path = [current_node]
    if visited_nodes is None:
        visited_nodes = set()

    # If we reached the target node, we found a physical path!
    if current_node == target_node and len(current_path) > 1:
        return [current_path]

    # Stop searching if we hit our depth limit
    # (Since current_path includes nodes, its length is: 2 * relations_count + 1)
    # For relation depth l, node length is l + 1
    current_relation_depth = (len(current_path) - 1) // 2
    if current_relation_depth >= max_depth:
        return []

    discovered_paths = []
    visited_nodes.add(current_node)

    # Explore outgoing edges from the current node
    if current_node in graph:
        for relation_id, neighbor_list in graph[current_node].items():
            for neighbor in neighbor_list:
                # Prevent infinite loops in cyclic graphs
                if neighbor not in visited_nodes or neighbor == target_node:
                    # Record BOTH the relation taken and the neighbor entity arrived at
                    new_path_extension = current_path + [relation_id, neighbor]
                    
                    new_paths = discover_paths_dfs_with_nodes(
                        graph=graph,
                        current_node=neighbor,
                        target_node=target_node,
                        max_depth=max_depth,
                        current_path=new_path_extension,
                        visited_nodes=visited_nodes.copy()
                    )
                    discovered_paths.extend(new_paths)

    return discovered_paths


def translate_full_path_to_strings(paths, id_to_entity, id_to_relation):
    """
    Converts a path list [entity_id, relation_id, entity_id...] 
    into a beautiful human-readable chain.
    """
    readable_paths = []
    for path in paths:
        path_segments = []
        for i, element in enumerate(path):
            if i % 2 == 0:
                # It's an Entity ID
                path_segments.append(f"({id_to_entity.get(element, f'Node_{element}')})")
            else:
                # It's a Relation ID
                path_segments.append(f" --[{id_to_relation.get(element, f'Rel_{element}')}]--> ")
        
        readable_paths.append("".join(path_segments))
    return sorted(readable_paths)


def main():
    # 1. Load the graph
    entities_path = 'dataset_uml/entities.txt'
    relations_path = 'dataset_uml/relations.txt'
    train_path = 'dataset_uml/train.txt'

    entity_to_id = graph_initializer.load_index_map(entities_path)
    relation_to_id = graph_initializer.load_index_map(relations_path)
    graph = graph_initializer.build_relation_adjacency_list(train_path, entity_to_id, relation_to_id)
    
    # Reverse maps to translate IDs back to text
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    print("\n[step3] Starting Detailed Path Discovery (with Intermediate Nodes)...")

    # 2. Pick a source and target entity pair to test
    source_entity = 'acquired_abnormality'
    target_entity = 'experimental_model_of_disease'
    
    source_id = entity_to_id.get(source_entity)
    target_id = entity_to_id.get(target_entity)

    if source_id is None or target_id is None:
        print(f"  [error] Source '{source_entity}' or target '{target_entity}' not found in entities.txt")
        return

    # 3. Discover all detailed paths up to path length l = 3
    max_path_length = 2
    print(f"  [info] Searching paths of length 1 to {max_path_length} between '{source_entity}' and '{target_entity}':")
    
    detailed_paths = discover_paths_dfs_with_nodes(graph, source_id, target_id, max_depth=max_path_length)

    # 4. Print Results
    print(f"\n[step3] Path Discovery Completed!")
    print(f"        Found {len(detailed_paths)} physical paths connecting these entities:")
    
    readable_results = translate_full_path_to_strings(detailed_paths, id_to_entity, id_to_relation)
    for index, path_str in enumerate(readable_results, 1):
        print(f"        Path #{index}: {path_str}")


if __name__ == "__main__":
    main()
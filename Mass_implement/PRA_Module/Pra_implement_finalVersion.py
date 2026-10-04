import os
from collections import defaultdict

# ------------------------------------------
# ---------- Step 1: Data Loading ----------
# ------------------------------------------
# Input = train.txt File
def load_dataset(file_path):
    """Loads triples from tab-separated text files."""
    triplets = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find dataset file at {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triplets.append((parts[0], parts[1], parts[2]))
    print(f"--- Successfully loaded {len(triplets)} triples.")
    return triplets



class MaSSPathTranslator:
    def __init__(self, triplets, threshold=0.9):
        
        # Initializes the path translator with triplets from the dataset.
        # Set threshold strictly to 0.9 to output only highly reliable paths.
        
        self.triplets = triplets
        self.threshold = threshold
        self.graph = defaultdict(lambda: defaultdict(set))
        self.relations = set()
        
        for h, r, t in triplets:
            self.graph[h][r].add(t)
            self.relations.add(r)
            
    def translate_relation(self, target_relation):

        # -------------------------------------------------------------------------------------------------------------- 
        # ---------- Step 2: Map each 2-Hop Path to its unique (source, target) pairs and track midddle nodes ----------
        # --------------------------------------------------------------------------------------------------------------
        # 2-Hop Path Mining:
        # Calculates: Confidence = (Pairs with direct target_relation) / (All pairs connected by r1 -> r2)

        # path_pairs meaning = { (Relation_1, Relation_2) : { (Source, Target) : {Set of Intermediate Nodes} } }
        path_pairs = defaultdict(lambda: defaultdict(set))
        
        print("--- Analyzing Knowledge Graph structure for 2-hop paths...")
        for s in self.graph:                            # pick a starting Source node "s"
            for r1 in self.graph[s]:                    # look at every relation outgoing from "s"
                for e_mid in self.graph[s][r1]:         # find the middle node e'
                    if e_mid not in self.graph:         # If the intermediate node is a "dead end" then -->> skip it and move to the next
                        continue
                    for r2 in self.graph[e_mid]:        # look at the second relation outgoing from the e'
                        for t in self.graph[e_mid][r2]: # find the final Target node

                            # Track unique (s, t) pairs and their connecting intermediate nodes
                            path_pairs[(r1, r2)][(s, t)].add(e_mid)
                            
        candidate_paths = []
        total_paths = len(path_pairs)
        print(f"--- Discovered {total_paths} unique 2-hop path templates.")

        # Save Path Templates to Outputs Folder ----------
        output_dir = r"C:\my-code\Robust Embedding\Mass_implement\outputs"
        output_file_path = os.path.join(output_dir, "Path_template_pra_output.txt")
        
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file_path, 'w', encoding='utf-8') as f_out:
            f_out.write(f"=== DISCOVERED 2-HOP PATH TEMPLATES (Total: {total_paths}) ===\n")
            f_out.write("Format: [r1] -> [r2]  |  (Unique connected entity pairs: count)\n")
            f_out.write("-" * 80 + "\n")
            
            # Sort the templates alphabetically by relation names for easy reading
            sorted_templates = sorted(path_pairs.keys())
            
            for index, (r1, r2) in enumerate(sorted_templates, 1):
                unique_pairs_count = len(path_pairs[(r1, r2)])
                f_out.write(f"Template {index:03d}:  {r1} -> {r2}  |  (Connected pairs: {unique_pairs_count})\n")
                
        print("--- Path Template saved successfully!")

        # -------------------------------------------------------------------------
        # ---------- Step 3: Calculate confidence for each path template ----------
        # ------------------------------------------------------------------------- 
        for (r1, r2), pairs in path_pairs.items():
            total_connected_pairs = len(pairs)  # Denominator: unique (s, t) pairs connected by path
            
            # Numerator: unique (s, t) pairs connected by path that have the target relation directly
            matching_pairs = []
            for (s, t), mid_nodes in pairs.items():
                if t in self.graph[s].get(target_relation, set()):
                    matching_pairs.append({
                        'source': s,
                        'intermediates': list(mid_nodes),
                        'target': t
                    })
            
            pos_target_count = len(matching_pairs)
            
            if pos_target_count > 0:
                confidence = pos_target_count / total_connected_pairs

                # ---------------------------------------------------------------------
                # ---------- Step 4: Confidence Greater than threshold (0.9) ----------
                # ---------------------------------------------------------------------
                if confidence > self.threshold:
                    candidate_paths.append(((r1, r2), confidence, matching_pairs))
                    
        # Sort templates by confidence score descending
        candidate_paths.sort(key=lambda x: x[1], reverse=True)
        return candidate_paths

    def find_two_hop_paths_for_triple(self, head, tail):
        """Return 2-hop relation templates and intermediate nodes from head to tail."""
        triple_paths = defaultdict(set)

        for r1, middle_nodes in self.graph.get(head, {}).items():
            for middle in middle_nodes:
                for r2, end_nodes in self.graph.get(middle, {}).items():
                    if tail in end_nodes:
                        triple_paths[(r1, r2)].add(middle)

        return triple_paths

    def translate_triple(self, head, target_relation, tail):

        """Return reliable target-relation templates that connect one H-R-T triple.
        Confidence is learned across the complete knowledge graph, then the
        templates are restricted to concrete paths from head to tail.
        """
        reliable_templates = self.translate_relation(target_relation)
        concrete_paths = self.find_two_hop_paths_for_triple(head, tail)
        matched_paths = []

        for path_template, confidence, _ in reliable_templates:
            intermediates = concrete_paths.get(path_template)
            if intermediates:
                matched_paths.append((path_template, confidence, sorted(intermediates)))

        return matched_paths


# ---------------------------------
# --- Execution & Testing Block ---
# ---------------------------------

if __name__ == "__main__":

    # Input Dataset Path
    train_file = r"C:\my-code\Robust Embedding\Mass_implement\dataset_uml\train.txt"
    
    try:
        clean_kg = load_dataset(train_file)
        
        # Target triple: change these values to query a different triple.
        TARGET_HEAD = "individual_behavior"
        TARGET_RELATION = "isa"
        TARGET_TAIL = "behavior"
        threshold = 0.9
        
        # We target strict rules with a confidence threshold > 0.9
        translator = MaSSPathTranslator(clean_kg, threshold=threshold)
        translated_paths = translator.translate_triple(
            TARGET_HEAD, TARGET_RELATION, TARGET_TAIL
        )
        direct_triple_exists = TARGET_TAIL in translator.graph.get(TARGET_HEAD, {}).get(
            TARGET_RELATION, set()
        )
        
        print(
            f"\n=== TARGET TRIPLE: "
            f"({TARGET_HEAD}, {TARGET_RELATION}, {TARGET_TAIL}) ===\n"
        )
        if not direct_triple_exists:
            print(
                "Note: the target triple is not a direct edge in the training graph. "
                "The paths below are potential PRA evidence.\n"
            )
        
        if not translated_paths:
            print(
                "No reliable 2-hop path connects the requested target triple "
                f"with the strict confidence threshold of > {threshold}."
            )
        else:
            for path_template, confidence, intermediates in translated_paths:
                print("-" * 85)
                print(f"Found Valid Path Template: {path_template[0]} -> {path_template[1]}")
                print(f"  Confidence Score: {confidence:.4f}")
                print("  Concrete Subgraph Traces for the requested triple (Max 5 Displayed):\n")
                
                for e_mid in intermediates[:5]:
                    print(f"    * Node({TARGET_HEAD}) "
                          f"---[{path_template[0]}]---> Node({e_mid}) "
                          f"---[{path_template[1]}]---> Node({TARGET_TAIL})\n")
                if len(intermediates) > 5:
                    print("    * ... (and more traces exist)")
                if direct_triple_exists:
                    print(
                        f" ** [Direct: {TARGET_HEAD} ---({TARGET_RELATION})---> "
                        f"{TARGET_TAIL}] \n"
                    )
                print("-" * 85)
                
    except FileNotFoundError as e:
        print(e)

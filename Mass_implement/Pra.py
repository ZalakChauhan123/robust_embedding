import random
from collections import defaultdict

class MaSSPathTranslator:
    def __init__(self, triplets, threshold=0.9):
        self.triplets = triplets
        self.threshold = threshold
        self.graph = defaultdict(lambda: defaultdict(set))
        self.relations = set()
        
        for h, r, t in triplets:
            self.graph[h][r].add(t)
            self.relations.add(r)
            
    def get_two_hop_paths(self):
        """
        MODIFIED: Now tracks the exact intermediate entity (e_mid) 
        alongside the structural co-occurrences.
        """
        path_cooccurrences = defaultdict(list) 
        
        for s in self.graph:
            for r1 in self.graph[s]:
                for e_mid in self.graph[s][r1]:
                    if e_mid not in self.graph:
                        continue
                    for r2 in self.graph[e_mid]:
                        for t in self.graph[e_mid][r2]:
                            for r_direct in self.graph[s]:
                                if t in self.graph[s][r_direct]:
                                    # Storing e_mid here so we can read it out later
                                    path_cooccurrences[(r1, r2)].append({
                                        'source': s,
                                        'intermediate': e_mid,
                                        'target': t,
                                        'r_direct': r_direct
                                    })
        return path_cooccurrences

    def translate_relation(self, target_relation):
        """
        MODIFIED: Returns the underlying trace samples along with the score
        so the user can inspect intermediate entity text.
        """
        path_cooccurrences = self.get_two_hop_paths()
        candidate_paths = []
        
        for (r1, r2), samples in path_cooccurrences.items():
            # Calculate metrics
            pos_target_count = sum(1 for s in samples if s['r_direct'] == target_relation)
            total_target_pair_count = len(samples)
            
            if total_target_pair_count > 0:
                confidence = pos_target_count / total_target_pair_count
                if confidence >= self.threshold:
                    # Pass the samples array forward to the output
                    candidate_paths.append(((r1, r2), confidence, samples))
                    
        candidate_paths.sort(key=lambda x: x[1], reverse=True)
        return candidate_paths

# --- Execution Block ---
if __name__ == "__main__":
    clean_kg = [
        ("Despicable Me", "defeat", "Nomadland"),
        ("Nomadland", "wonNomination", "Oscar"),
        ("Despicable Me", "won", "Oscar"), 
        
        ("The King's Speech", "defeat", "The Social Network"),
        ("The Social Network", "wonNomination", "Oscar"),
        ("The King's Speech", "won", "Oscar"), 
    ]
    
    TARGET = "isa"
    translator = MaSSPathTranslator(clean_kg, threshold=0.9)
    translated_paths = translator.translate_relation(TARGET)
    
    print(f"=== PRA PATH TRANSLATION FOR TARGET RELATION: '{TARGET}' ===\n")
    
    for path_template, confidence, samples in translated_paths:
        print(f"Found Valid Path Template: {path_template[0]} -> {path_template[1]}")
        print(f"  Confidence Score: {confidence:.2f}")
        print("  Concrete Subgraph Traces Found in KG:")
        
        # Loop through samples to extract the source, intermediate entity, and target
        for sample in samples:
            if sample['r_direct'] == TARGET:
                print(f"    * Node({sample['source']}) "
                      f"--[{path_template[0]}]--> Node({sample['intermediate']}) "
                      f"--[{path_template[1]}]--> Node({sample['target']})")
        print("-" * 60)
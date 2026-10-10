"""
-- Input = UMLS KG. 
-- Output = trained TransE checkpoint containing embeddings. 
-- These embeddings allow Entity_Selection.py to compute the relation similarity, triple plausibility, and lower-bound scores that Paper's Section 4.4.1 requires.
"""
# Import Modules
from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MASS_DIR = Path(__file__).resolve().parents[1]


# Read entity.txt & relation.txt
def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


# Read UML training triples (train.txt)
def read_triples(path: Path) -> list[tuple[str, str, str]]:
    triples = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            columns = line.rstrip("\r\n").split("\t")
            if len(columns) != 3:
                raise ValueError(f"Bad triple at {path}:{line_no}: {line!r}")
            triples.append(tuple(columns))
    return triples

# TransE -- Embedding Model
class TransE(nn.Module):

    # Create and initialize embedding matrices
    def __init__(self, num_entities: int, num_relations: int, dim: int = 64):
        super().__init__()
        self.entity = nn.Embedding(num_entities, dim) # nn.Embedding() = Numerical Vector representation of Entities
        self.relation = nn.Embedding(num_relations, dim) # # nn.Embedding() = Numerical Vector representation of Relations
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    # Calculate TransE distance for triples
    def distance(self, heads: torch.Tensor, relations: torch.Tensor,
                 tails: torch.Tensor) -> torch.Tensor:

        # Implement TransE Formula = h+r = t
        return torch.linalg.vector_norm(
            self.entity(heads) + self.relation(relations) - self.entity(tails),
            ord=2, dim=-1,
        )
    # Convert distance into plausibility score (TransE Model Score)
    def score(self, heads: torch.Tensor, relations: torch.Tensor,
              tails: torch.Tensor) -> torch.Tensor:
        return -self.distance(heads, relations, tails)

"""
Wrapper around the trained TransE model that makes calculating relation similarity & triple plausibility.

"""
class Surrogate:
    # Wrap trained TransE
    def __init__(self, model: TransE, entity_to_id: dict[str, int],
                 relation_to_id: dict[str, int]):
        self.model = model.cpu().eval()
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id

    @torch.no_grad()

    # Calculate MaSS relation similarity
    def relation_score(self, target: str, r1: str, r2: str) -> float:
        emb = self.model.relation.weight
        target_vector = emb[self.relation_to_id[target]]
        first = emb[self.relation_to_id[r1]]
        second = emb[self.relation_to_id[r2]]
        return float(-torch.linalg.vector_norm(target_vector - first - second, ord=2))

    @torch.no_grad()

    # Calculate triple plausibility
    def triple_score(self, head: str, relation: str, tail: str) -> float:
        heads = torch.tensor([self.entity_to_id[head]])
        relations = torch.tensor([self.relation_to_id[relation]])
        tails = torch.tensor([self.entity_to_id[tail]])
        return float(self.model.score(heads, relations, tails).item())

    @classmethod
    # Load the saved trained model (Without retrain again)
    def load(cls, checkpoint_path: Path) -> "Surrogate":
        # Load only checkpoints you created/trust; torch checkpoints are not untrusted input.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = TransE(
            len(checkpoint["entity_to_id"]),
            len(checkpoint["relation_to_id"]),
            checkpoint["embedding_dim"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        return cls(model, checkpoint["entity_to_id"], checkpoint["relation_to_id"])


"""
Train TransE -- training Process (Step A to F)
-----------
Parameters
-----------
epochs = 300 --- Train over the dataset 300 times
dim = 64 --- Each embedding contains 64 values
batch_size = 256 --- Process up to 256 positive triples at a time
seed = 42 --- Make random initialization and sampling reproducible
lr = 0.005 --- Learning rate for parameter updates
margin = 1.0 --- Required separation between positive and negative distances
"""
def train(checkpoint_path: Path, 
          epochs: int = 300, 
          dim: int = 64,
          batch_size: int = 256, 
          seed: int = 42, 
          lr: float = 0.005,
          margin: float = 1.0) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = MASS_DIR / "dataset_uml"
    entities = read_names(dataset / "entities.txt")
    relations = read_names(dataset / "relations.txt")
    entity_to_id = {name: index for index, name in enumerate(entities)}
    relation_to_id = {name: index for index, name in enumerate(relations)}
    raw_triples = read_triples(dataset / "train.txt")

    # STEP A -- CONVERT TRIPLES INTO NUMERICAL IDS
    triples = torch.tensor([
        (entity_to_id[h], relation_to_id[r], entity_to_id[t])
        for h, r, t in raw_triples
    ], dtype=torch.long)

    # STEP B -- CREATE MODEL & OPTIMIZER
    model = TransE(len(entities), len(relations), dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training TransE: {len(triples)} triples, {len(entities)} entities, "
          f"{len(relations)} relations")

    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(triples))
        total_loss = 0.0

        for batch_indices in indices.split(batch_size):
            positive = triples[batch_indices]

            # STEP C -- CREATE NEGETIVE TRIPLETS
            negative = positive.clone()
            corrupt_head = torch.rand(len(negative)) < 0.5
            replacement = torch.randint(len(entities), (len(negative),))
            negative[corrupt_head, 0] = replacement[corrupt_head]
            negative[~corrupt_head, 2] = replacement[~corrupt_head]

            # STEP D -- CALCULATE TRAINING LOSS (Loss = max(0, margin + d(v+) - d(v-) ))
            # Why needed -- V+ Triplets comes closer & V- Triplets become further away
            positive_distance = model.distance(
                positive[:, 0], positive[:, 1], positive[:, 2]
            )
            negative_distance = model.distance(
                negative[:, 0], negative[:, 1], negative[:, 2]
            )

            loss = F.relu(
                margin + positive_distance - negative_distance
            ).mean()

            # STEP E -- BACKPROPOGATION
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.entity.weight.copy_(
                    F.normalize(model.entity.weight, p=2, dim=1)
                )

            total_loss += loss.item() * len(batch_indices)

        # OUTSIDE BATCH LOOP -- INSIDE EPOCH LOOP
        epoch_loss = total_loss / len(triples)
        loss_history.append(epoch_loss)

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:4d}/{epochs} - loss {epoch_loss:.4f}"
            )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # STEP F -- SAVE TRAINED EMBEDDINGS
    # Output = transE_umls.pt
    torch.save({
        "state_dict": model.state_dict(),
        "entity_to_id": entity_to_id,
        "relation_to_id": relation_to_id,
        "embedding_dim": dim,
    }, checkpoint_path)
    print(f"Saved surrogate: {checkpoint_path}")
    with (checkpoint_path.parent / "training_loss.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "training_loss"])
        writer.writerows(enumerate(loss_history, start=1))
    print(f"Saved loss history: {checkpoint_path.parent / 'training_loss.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--margin", type=float, default=1.0)

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MASS_DIR / "outputs" / "Path_Generation" / "transe_umls.pt"
    )

    args = parser.parse_args()

    train(
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        dim=args.dim,
        batch_size=args.batch_size,
        seed=args.seed,
        lr=args.lr,
        margin=args.margin
    )

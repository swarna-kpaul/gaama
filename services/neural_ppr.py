"""Neural adaptive PPR weight predictor.

Architecture (Gap Regression):
    h = ReLU(W1 @ embedding + b1)   # [1536] -> [hidden_dim]
    gap = W2 @ h + b2               # [hidden_dim] -> [1]
    ppr_weight = 1.0 if gap > 0 else 0.01

The model predicts the reward gap (r_ppr1 - r_ppr0). If positive, PPR helps
and we use ppr_weight=1.0. If negative, PPR hurts and we use ppr_weight=0.01.

Uses PyTorch for training; pure Python forward for lightweight SDK inference.
"""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model (JSON-serialisable, pure-Python inference)
# ---------------------------------------------------------------------------

@dataclass
class NeuralPPRModel:
    """Gap regression MLP: predicts reward gap (r_ppr1 - r_ppr0) from query embedding.

    Architecture:
        h = ReLU(W1 @ embedding + b1)    # [embed_dim] -> [hidden_dim]
        gap = W2 @ h + b2                 # [hidden_dim] -> [1]
        ppr_weight = 1.0 if gap > 0 else 0.01
    """
    embed_dim: int = 1536
    hidden_dim: int = 32

    W1: List[List[float]] = field(default_factory=list)
    b1: List[float] = field(default_factory=list)
    W2: List[float] = field(default_factory=list)
    b2: float = 0.0

    ppr_high: float = 1.0
    ppr_low: float = 0.01

    def __post_init__(self):
        if not self.W1:
            self._init_weights()

    def _init_weights(self):
        scale1 = math.sqrt(2.0 / (self.embed_dim + self.hidden_dim))
        self.W1 = [
            [random.gauss(0, scale1) for _ in range(self.embed_dim)]
            for _ in range(self.hidden_dim)
        ]
        self.b1 = [0.0] * self.hidden_dim
        scale2 = math.sqrt(2.0 / (self.hidden_dim + 1))
        self.W2 = [random.gauss(0, scale2) for _ in range(self.hidden_dim)]
        self.b2 = 0.0

    def forward(self, embedding: Sequence[float]) -> float:
        """Predict reward gap. Pure Python -- no torch needed at inference."""
        z1 = [
            sum(wj * ej for wj, ej in zip(self.W1[j], embedding)) + self.b1[j]
            for j in range(self.hidden_dim)
        ]
        h1 = [max(0.0, v) for v in z1]
        gap = sum(wj * hj for wj, hj in zip(self.W2, h1)) + self.b2
        return gap

    def predict(self, query_embedding: Sequence[float]) -> float:
        """Predict optimal ppr_weight: high if gap > 0, low otherwise."""
        gap = self.forward(query_embedding)
        return self.ppr_high if gap > 0 else self.ppr_low

    def save(self, path: Path) -> None:
        data = {
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "ppr_high": self.ppr_high,
            "ppr_low": self.ppr_low,
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NeuralPPRModel":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


# ---------------------------------------------------------------------------
# Trainer (PyTorch)
# ---------------------------------------------------------------------------

class NeuralPPRTrainer:
    """Train NeuralPPRModel via gap regression using PyTorch."""

    def __init__(
        self,
        model: NeuralPPRModel,
        lr: float = 0.0005,
        weight_decay: float = 0.01,
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    @staticmethod
    def _build_torch_model(embed_dim: int, hidden_dim: int):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def train(
        self,
        data: List[Tuple[List[float], float]],
        epochs: int = 400,
        batch_size: int = 64,
        log_every: int = 100,
        val_data: Optional[List[Tuple[List[float], float]]] = None,
        early_stopping_patience: int = 50,
        diff_weight: float = 10.0,
        tie_weight: float = 0.5,
    ) -> Dict:
        """Train on (embedding, gap) pairs.

        Args:
            data: list of (embedding, reward_gap) where gap = r_ppr1 - r_ppr0
            diff_weight: weight multiplier for examples with non-zero gap
            tie_weight: weight for examples with zero gap
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        if not data:
            return {"epochs": 0, "final_loss": 0.0}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = self._build_torch_model(self.model.embed_dim, self.model.hidden_dim).to(device)

        embs_t = torch.tensor([d[0] for d in data], dtype=torch.float32, device=device)
        gaps_t = torch.tensor([d[1] for d in data], dtype=torch.float32, device=device)
        weights_t = torch.tensor(
            [diff_weight * abs(d[1]) if abs(d[1]) > 0 else tie_weight for d in data],
            dtype=torch.float32, device=device,
        )

        dataset = TensorDataset(embs_t, gaps_t, weights_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_data:
            ve = torch.tensor([d[0] for d in val_data], dtype=torch.float32, device=device)
            vg = torch.tensor([d[1] for d in val_data], dtype=torch.float32, device=device)
            vw = torch.tensor(
                [diff_weight * abs(d[1]) if abs(d[1]) > 0 else tie_weight for d in val_data],
                dtype=torch.float32, device=device,
            )
            val_loader = DataLoader(TensorDataset(ve, vg, vw), batch_size=256, shuffle=False)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history = []

        for epoch in range(epochs):
            net.train()
            epoch_loss = 0.0
            n_batches = 0
            for emb_b, gap_b, w_b in loader:
                optimizer.zero_grad()
                pred = net(emb_b).squeeze(-1)
                loss = (w_b * (pred - gap_b) ** 2).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            entry = {"epoch": epoch + 1, "train_loss": avg_loss}

            if val_loader is not None:
                net.eval()
                val_loss = 0.0
                vn = 0
                n_correct = 0
                n_diff = 0
                with torch.no_grad():
                    for emb_b, gap_b, w_b in val_loader:
                        pred = net(emb_b).squeeze(-1)
                        val_loss += (w_b * (pred - gap_b) ** 2).mean().item()
                        vn += 1
                        mask = gap_b != 0
                        if mask.sum() > 0:
                            n_correct += ((pred > 0) == (gap_b > 0))[mask].sum().item()
                            n_diff += mask.sum().item()
                entry["val_loss"] = val_loss / max(1, vn)
                entry["val_sign_acc"] = n_correct / max(1, n_diff)

                if entry["val_loss"] < best_val_loss:
                    best_val_loss = entry["val_loss"]
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if log_every:
                            print(f"  Early stopping at epoch {epoch+1} "
                                  f"(best val_loss={best_val_loss:.6f} at epoch {epoch+1-patience_counter})")
                        break

            history.append(entry)

            if log_every and (epoch + 1) % log_every == 0:
                msg = f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}"
                if "val_loss" in entry:
                    msg += f"  val_loss={entry['val_loss']:.6f}"
                if "val_sign_acc" in entry:
                    msg += f"  sign_acc={entry['val_sign_acc']:.1%}"
                print(msg)

        if best_state is not None:
            net.load_state_dict(best_state)

        # Copy weights back to model
        with torch.no_grad():
            state = net.state_dict()
            # Sequential: 0=Linear, 1=ReLU, 2=Dropout, 3=Linear
            self.model.W1 = state["0.weight"].cpu().tolist()
            self.model.b1 = state["0.bias"].cpu().tolist()
            self.model.W2 = state["3.weight"].cpu().squeeze(0).tolist()
            self.model.b2 = float(state["3.bias"].cpu().item())

        return {
            "epochs": len(history),
            "final_loss": history[-1]["train_loss"] if history else 0.0,
            "final_val_loss": history[-1].get("val_loss") if history else None,
            "history": history,
        }

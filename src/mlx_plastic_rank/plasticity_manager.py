from typing import List, Dict, Optional
import mlx.core as mx
import mlx.nn as nn

from .lowrank import RankLayer
from .utils import get_logger
from .rank_select import choose_rank


class PlasticityManager:
    """Adaptive rank controller for ``RankLayer`` modules.

    Orchestrates growth, pruning, and waking of low-rank factors based on
    validation loss dynamics and rank selection heuristics.

    Parameters
    - model: nn.Module
        Model containing one or more ``RankLayer`` modules.
    - delta: float = 0.01
        Plateau threshold for |val_t - val_{t-1}|; below this, enter plastic phase.
    - tol: float = 1e-4
        Tolerance passed to pruning logic and some heuristics.
    - strategy: str = "stable"
        Rank selection strategy: "stable" energy cutoff or "theorem".
    - target_compression: float = 0.9
        Fraction of spectral energy to retain when choosing rank (higher keeps more).
    - gamma: float = 1e-3
        Reactivation threshold; if loss worsens by > gamma after a shrink, wake sleepers.
    - log_path: Optional[str] = "out/plasticity.jsonl"
        Where to append JSONL events; set to None to disable.
    """
    def __init__(
        self,
        model: nn.Module,
        delta: float = 0.01,
        tol: float = 1e-4,
        strategy: str = "stable",
        target_compression: float = 0.9,
        gamma: float = 1e-3,
        log_path: Optional[str] = "out/plasticity.jsonl",
    ):
        self.model = model
        self.delta = delta
        self.tol = tol
        self.strategy = strategy
        self.target_compression = target_compression
        self.gamma = gamma
        self.log_path = log_path
        self.layers: List[RankLayer] = [m for m in model.modules() if isinstance(m, RankLayer)]
        self.history: List[float] = []
        self._last_action: Dict[int, str] = {}

    def step(self, metrics: Dict[str, float]):
        val_loss = float(metrics["val_loss"])
        self.history.append(val_loss)
        if len(self.history) < 2:
            return
        if abs(self.history[-1] - self.history[-2]) < self.delta:
            self._plastic_phase()
        # Reactivation path: if last action was shrink and loss worsened beyond gamma
        if len(self.history) >= 2 and (self.history[-1] - self.history[-2]) > self.gamma:
            for idx, lyr in enumerate(self.layers):
                if self._last_action.get(idx) == "shrink" and lyr.sleep_dict:
                    # wake last K sleepers (here K=1 for simplicity)
                    last_idx = max(lyr.sleep_dict.keys())
                    lyr.wake_rank(last_idx)
                    self._log_event(
                        layer_name=f"layer_{idx}",
                        r0=lyr.rank - 1,
                        r_star=lyr.rank,
                        residual=-1.0,
                        action="wake",
                        sleep_bytes=self._sleep_bytes(lyr),
                        val_before=self.history[-2],
                        val_after=self.history[-1],
                    )

    def _plastic_phase(self):
        logger = get_logger()
        for idx, lyr in enumerate(self.layers):
            # crude LRP proxy: gradient magnitude of W0
            g = mx.grad(lambda W: mx.square(W).sum())(lyr.W0)
            score = float(mx.abs(g).mean())
            # Effective weight for choosing rank
            W = lyr.W0
            if lyr.rank > 0:
                W = W + lyr.U.T @ (lyr.V * lyr.S[:, None])
            r0 = lyr.rank
            r_star, residual = choose_rank(W, self.target_compression, self.strategy)
            action = None
            if r_star > r0:
                lyr.add_rank(r_star - r0)
                action = "grow"
            elif r_star < r0:
                before_bytes = self._sleep_bytes(lyr)
                lyr.prune_to_rank(r_star)
                action = "shrink"
                after_bytes = self._sleep_bytes(lyr)
                logger.info(
                    f"Plastic: layer_{idx} shrink from r0={r0} to r*={r_star}; sleep +={after_bytes - before_bytes} bytes"
                )
            if action:
                self._log_event(
                    layer_name=f"layer_{idx}",
                    r0=r0,
                    r_star=r_star,
                    residual=residual,
                    action=action,
                    sleep_bytes=self._sleep_bytes(lyr),
                    val_before=self.history[-2],
                    val_after=self.history[-1],
                )
                self._last_action[idx] = action

    def compress_dict_size(self) -> int:
        total = 0
        for lyr in self.layers:
            for entry in lyr.sleep_dict.values():
                # entry = (q_u, mn_u, sc_u, s, q_v, mn_v, sc_v)
                q_u, _, _, _, q_v, _, _ = entry
                total += int(q_u.size + q_v.size)
        return total  # bytes (uint8 elements)

    def _sleep_bytes(self, lyr: RankLayer) -> int:
        total = 0
        for entry in lyr.sleep_dict.values():
            q_u, _, _, _, q_v, _, _ = entry
            total += int(q_u.size + q_v.size)
        return total

    def _log_event(
        self,
        layer_name: str,
        r0: int,
        r_star: int,
        residual: float,
        action: str,
        sleep_bytes: int,
        val_before: float,
        val_after: float,
    ) -> None:
        import json, os, datetime

        if not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        ts = datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")
        evt = {
            "ts": ts,
            "layer": layer_name,
            "r0": int(r0),
            "r_star": int(r_star),
            "residual": float(residual),
            "action": action,
            "sleep_bytes": int(sleep_bytes),
            "val_loss_before": float(val_before),
            "val_loss_after": float(val_after),
            "strategy": self.strategy,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt) + "\n")

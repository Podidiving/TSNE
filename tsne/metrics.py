import numpy as np
import torch
from typing import Callable, List
from math import fabs


class Calculator:
    @classmethod
    def perplexity(cls, probs: torch.Tensor) -> float:
        probs = probs[
            torch.logical_not(
                torch.isclose(
                    probs, torch.FloatTensor([0.0], device=probs.device)
                )
            )
        ]
        return 2 ** (-torch.sum(probs * torch.log2(probs)))

    @classmethod
    def conditional_probs(
        cls, square_distances: torch.Tensor, idx: int, sigma: float
    ) -> torch.Tensor:
        numenator = torch.exp(-square_distances[idx] / (2 * (sigma ** 2)))
        numenator[idx] = 0
        return numenator / numenator.sum()

    @classmethod
    def joint_probs_q(
        cls,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        square_distances = torch.cdist(Y, Y).pow(2)
        
        qs = (1 + square_distances).pow(-1)
        ind = np.diag_indices(qs.shape[0])
        qs[ind[0], ind[1]] *= 0
        ind = np.triu_indices(n=qs.shape[0])
        qs = qs[ind]
        qs = qs / qs.sum()

        return qs

    @classmethod
    def kl_distance(cls, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        zeros_mask = torch.isclose(
            P, torch.FloatTensor([0.0], device=P.device)
        )
        P, Q = P[~zeros_mask], Q[~zeros_mask]
        zeros_mask = torch.isclose(
            Q, torch.FloatTensor([0.0], device=Q.device)
        )
        P, Q = P[~zeros_mask], Q[~zeros_mask]
        
        kl_distance = (P * (torch.log(P) - torch.log(Q))).sum()
        return kl_distance

    @classmethod
    def sigmas(
        cls, distances: torch.Tensor, perplexity: float
    ) -> List[float]:
        return [
            cls.__binary_search(
                lambda sigma: cls.perplexity(
                    cls.conditional_probs(
                        distances, idx=idx, sigma=sigma
                    )
                ),
                target=perplexity,
            )
            for idx in range(len(distances))
        ]

    @classmethod
    def __binary_search(
        cls,
        f: Callable[[float], float],
        target: float,
        start: float = 1.0,
        tol: float = 1e-5,
        n_iter: int = 100,
    ) -> float:

        left = float("-inf")
        right = float("inf")
        mid = start

        for _ in range(n_iter):
            diff = f(mid) - target

            if fabs(diff) < tol:
                return mid

            if diff < 0:
                left = mid
                if right == float("inf"):
                    mid *= 2
                else:
                    mid = (mid + right) / 2
            else:
                right = mid
                if left == float("-inf"):
                    mid /= 2
                else:
                    mid = (mid + left) / 2
        return mid

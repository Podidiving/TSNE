from sklearn.base import TransformerMixin
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
from loguru import logger

from .metrics import Calculator


class TSNE(TransformerMixin):
    def __init__(
        self,
        n_components: int = 3,
        perplexity: float = 30.0,
        n_iter: int = 1000,
        lr: float = 1.0,
        momentum: float = 0.9,
        log_interval: int = 100,
        seed: Optional[int] = 42,
    ):
        torch.manual_seed(seed)

        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.momentum = momentum
        self.lr = lr
        self.log_interval = log_interval
        self.device = (
            torch.device("cuda:0")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.joint_prob = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        with torch.no_grad():
            X_ = torch.from_numpy(X).float().to(self.device)
            square_distances = torch.cdist(X_, X_).pow(2)

            sigmas = Calculator.sigmas(square_distances, self.perplexity)

            conditional = [
                Calculator.conditional_probs(
                    square_distances, idx=idx, sigma=sigma
                )[None, :]
                for idx, sigma in enumerate(sigmas)
            ]
            conditional = torch.cat(conditional, 0)

            joint_probs = conditional[np.triu_indices(n=conditional.shape[0])]

            self.joint_prob = joint_probs.clone()
            return self

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        Y = torch.randn(
            X.shape[0],
            self.n_components,
            requires_grad=True,
            dtype=torch.float,
            device=self.device,
        )

        opt = torch.optim.SGD(
            params=[Y],
            lr=self.lr,
            momentum=self.momentum,
        )
        scheduler = ReduceLROnPlateau(opt)

        for i in range(self.n_iter):
            opt.zero_grad()

            kl = Calculator.kl_distance(
                self.joint_prob, Calculator.joint_probs_q(Y)
            )

            kl.backward()
            opt.step()
            scheduler.step(kl)

            if not (i + 1) % self.log_interval:
                logger.info(
                    "{}% : KL - {}",
                    round((i / self.n_iter) * 100, 3),
                    kl.cpu().detach().item(),
                )

        return Y.detach().numpy()

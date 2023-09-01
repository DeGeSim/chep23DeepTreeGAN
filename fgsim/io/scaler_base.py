from pathlib import Path
from typing import Callable, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from fgsim.config import conf


class ScalerBase:
    def __init__(
        self,
        files: List[Path],
        len_dict: Dict,
        read_chunk: Callable,
        transform_wo_scaling: Callable,
        transfs,
    ) -> None:
        self.files = files
        self.len_dict = len_dict
        self.transfs = transfs
        self.read_chunk = read_chunk
        self.transform_wo_scaling = transform_wo_scaling
        self.scalerpath = Path(conf.path.dataset_processed) / "scaler.gz"
        if not self.scalerpath.is_file():
            self.save_scaler()
        self.transfs = joblib.load(self.scalerpath)

    def save_scaler(self):
        assert self.len_dict[self.files[0]] >= conf.loader.scaling_fit_size
        chk = self.read_chunk(
            [(Path(self.files[0]), 0, conf.loader.scaling_fit_size)]
        )
        event_list = [self.transform_wo_scaling(e) for e in chk]

        # The features need to be converted to numpy immediatly
        # otherwise the queuflow afterwards doesnt work
        pcs = np.vstack([e.x.clone().numpy() for e in event_list])
        if hasattr(event_list[0], "mask"):
            mask = np.hstack([e.mask.clone().numpy() for e in event_list])
            pcs = pcs[mask]

        self.plot_scaling(pcs)

        assert pcs.shape[1] == conf.loader.n_features
        pcs = np.hstack(
            [
                transf.fit_transform(arr.reshape(-1, 1))
                for arr, transf in zip(pcs.T, self.transfs)
            ]
        )
        self.plot_scaling(pcs, True)

        joblib.dump(self.transfs, self.scalerpath)

    def transform(self, pcs: np.ndarray):
        assert len(pcs.shape) == 2
        assert pcs.shape[1] == conf.loader.n_features
        return np.hstack(
            [
                transf.transform(arr.reshape(-1, 1))
                for arr, transf in zip(pcs.T, self.transfs)
            ]
        )

    def inverse_transform(self, pcs: torch.Tensor):
        assert pcs.shape[-1] == conf.loader.n_features
        orgshape = pcs.shape
        dev = pcs.device
        pcs = pcs.to("cpu").detach().reshape(-1, conf.loader.n_features).numpy()

        t_stacked = np.hstack(
            [
                transf.inverse_transform(arr.reshape(-1, 1))
                for arr, transf in zip(pcs.T, self.transfs)
            ]
        )
        return torch.from_numpy(t_stacked.reshape(*orgshape)).float().to(dev)

    def plot_scaling(self, pcs, post=False):
        for k, v in zip(conf.loader.x_features, pcs.T):
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.hist(v, bins=500)
            fig.savefig(
                Path(conf.path.dataset_processed) / f"{k}_post.png"
                if post
                else Path(conf.path.dataset_processed) / f"{k}_pre.png"
            )
            plt.close(fig)

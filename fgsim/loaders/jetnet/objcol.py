from pathlib import Path
from typing import List, Tuple

import torch
from jetnet.datasets import JetNet
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch_geometric.data import Data

from fgsim.config import conf
from fgsim.io import FileManager, ScalerBase

jn_dict = {}


def get_jn(fn):
    fn = str(fn)
    if fn not in jn_dict:
        jn_dict[fn] = JetNet.getData(
            jet_type=fn, data_dir=Path(conf.loader.dataset_path).expanduser()
        )
    return jn_dict[fn]


def path_to_len(fn: Path) -> int:
    return get_jn(fn)[0].shape[0]


file_manager = FileManager(path_to_len, files=[Path(conf.loader.jettype)])


def readpath(
    fn: Path,
    start: int,
    end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    particle_data, jet_data = get_jn(fn)
    res = (
        torch.tensor(jet_data[start:end], dtype=torch.float),
        torch.tensor(particle_data[start:end], dtype=torch.float),
    )
    return res


def read_chunks(
    chunks: List[Tuple[Path, int, int]]
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    chunks_list = []
    for chunk in chunks:
        chunks_list.append(readpath(*chunk))
    res = (
        torch.concat([e[0] for e in chunks_list]),
        torch.concat([e[1] for e in chunks_list]),
    )
    return [(res[0][ievent], res[1][ievent]) for ievent in range(len(res[1]))]


def contruct_graph_from_row(chk: Tuple[torch.Tensor, torch.Tensor]) -> Data:
    y, x = chk
    res = Data(
        x=x[x[..., 3].bool(), :3].reshape(-1, 3),
        y=y.reshape(1, -1),
        n_pointsv=x[..., 3].sum(),
    )
    return res


scaler = ScalerBase(
    files=file_manager.files,
    len_dict=file_manager.file_len_dict,
    transfs=[
        StandardScaler(),
        StandardScaler(),
        PowerTransformer(method="box-cox", standardize=True),
    ],
    read_chunk=read_chunks,
    transform_wo_scaling=contruct_graph_from_row,
)
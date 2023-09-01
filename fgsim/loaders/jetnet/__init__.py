"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""

import sys

from ..base_dataset import BaseDS
from .objcol import contruct_graph_from_row, file_manager, read_chunks


class JetNetDS(BaseDS):
    def __init__(self):
        super().__init__(file_manager)

    def _chunk_to_batch(self, chunks):
        chunks = read_chunks(chunks)
        batch = Batch.from_data_list(
            [contruct_graph_from_row([ey, ex]) for ey, ex in chunks]
        )
        batch.x = scaler.transform(batch.x, "x")
        batch.y = scaler.transform(batch.y, "y")
        return batch


if "pytest" not in sys.modules:
    from torch_geometric.data import Batch

    # from fgsim.io import LoaderInfo
    from .objcol import scaler  # file_manager

    # from .seq import process_seq, shared_batch_size, shared_postprocess_switch
    # loader = LoaderInfo(
    #     file_manager=file_manager,
    #     scaler=scaler,
    #     process_seq=process_seq,
    #     shared_postprocess_switch=shared_postprocess_switch,
    #     shared_batch_size=shared_batch_size,
    #     Batch=Batch,
    # )

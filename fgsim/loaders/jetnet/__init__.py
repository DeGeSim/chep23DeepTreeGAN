"""
Here steps for reading the root files and processing the hit list \
to graphs are definded. `process_seq` is the function that \
should be passed the qfseq.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from fgsim.config import conf

from fgsim.io.chunks import ChunkManager

from torch.multiprocessing import Pool
from fgsim.monitoring import logger
from .objcol import file_manager, read_chunks, contruct_graph_from_row
from tqdm import tqdm


class BaseDS:
    def __init__(self):
        self.chunk_manager = ChunkManager(file_manager)

    def _provide_batches(self, dsname):
        attr_name = f"_{dsname}_batches"
        pickle_path = Path(conf.path.dataset_processed) / f"{dsname}.pt"
        chunks = getattr(self.chunk_manager, f"{dsname}_chunks")
        return self._provide_batches_args(attr_name, pickle_path, chunks)

    def _provide_batches_args(self, attr_name, pickle_path, chunks):
        if not hasattr(self, attr_name):
            logger.debug(f"{attr_name} batches not loaded")
            if pickle_path.is_file():
                batch_list = torch.load(
                    pickle_path, map_location=torch.device("cpu")
                )
            else:
                batch_list = self.__process_ds(chunks)
                torch.save(batch_list, pickle_path)
            setattr(self, attr_name, batch_list)
        return getattr(self, attr_name)

    @property
    def training_batches(self):
        return self._provide_batches("training")

    @property
    def validation_batches(self):
        return self._provide_batches("validation")

    @property
    def testing_batches(self):
        return self._provide_batches("testing")

    @property
    def eval_batches(self):
        batch_list = self.validation_batches + self.testing_batches
        if conf.loader.eval_glob is None:
            return batch_list
        rest_eval_path = Path(conf.path.dataset_processed) / "rest_eval.pt"
        batch_list += self._provide_batches_args(
            "_rest_eval_batches",
            rest_eval_path,
            self.chunk_manager.rest_eval_chunks,
        )
        return batch_list

    def __process_ds(self, chunks_list):
        batch_list = [self._chunk_to_batch(e) for e in tqdm(chunks_list)]
        # batch_list = []
        # with Pool(2) as p:
        #     with tqdm(total=len(chunks_list)) as pbar:
        #         # for b in p.imap_unordered(self._chunk_to_batch, chunks_list):
        #         for b in (self._chunk_to_batch(e) for e in chunks_list):
        #             batch_list.append(b.clone())
        #             pbar.update()

        return batch_list

    def _chunk_to_batch(self, chunks):
        raise NotImplementedError

    def queue_epoch(self, n_skip_events=0) -> None:
        pass

    def __iter__(self):
        return iter(self.training_batches)


class JetNetDS(BaseDS):
    def __init__(self):
        super().__init__()

    def _chunk_to_batch(self, chunks):
        print("start")
        chunks = read_chunks(chunks)
        batch = Batch.from_data_list(
            [contruct_graph_from_row([ey, ex]) for ey, ex in chunks]
        )
        batch.x = torch.from_numpy(scaler.transform(batch.x.numpy())).float()
        print("end")
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

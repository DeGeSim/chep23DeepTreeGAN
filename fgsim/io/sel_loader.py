import importlib

from fgsim.config import conf

# from typing import List, Type


# from fgsim.io.loader import LoaderInfo

# Import the specified processing sequence
scaler = importlib.import_module(f"fgsim.datasets.{conf.dataset_name}").scaler


# file_manager = loader_info.file_manager
if not conf.loader.preprocess_training:
    scaler.fit()
# process_seq = loader_info.process_seq
# shared_postprocess_switch = loader_info.shared_postprocess_switch
# Batch: Type = loader_info.Batch

# DataSetType = List[Batch]
# files = loader_info.file_manager.files
# len_dict = loader_info.file_manager.file_len_dict

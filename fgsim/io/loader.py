from dataclasses import dataclass
from typing import List, Type, Union

from .file_manager import FileManager
from .scaler_base import ScalerBase

raise NotImplementedError
@dataclass
class LoaderInfo:
    file_manager: FileManager
    scaler: ScalerBase
    shared_postprocess_switch: Value
    shared_batch_size: Value
    Batch: Type

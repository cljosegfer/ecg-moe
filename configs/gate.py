
import torch

from dataclasses import dataclass, field
from typing import Any

@dataclass
class LoadDataConfig:
    hdf5_path: str = '/home/josegfer/code/code15/code15.h5'
    metadata_path: str = '/home/josegfer/code/code15/exams.csv'
    batch_size: int = 128
    tracing_col: str = "tracings"
    exam_id_col: str = "exam_id"
    patient_id_col: str = "patient_id"
    output_col: list = field(
        default_factory = lambda: [
            "1dAVb",
            "RBBB",
            "LBBB",
            "SB",
            "AF",
            "ST",
        ]
    )
    tracing_dataset_name: str = "tracings"
    exam_id_dataset_name: str = "exam_id"
    test_size: float = 0.05
    val_size: float = 0.1
    random_seed: int = 0
    data_dtype: Any = torch.float32
    output_dtype: Any = torch.long
    block_classes: list = field(
        default_factory = lambda: [
            "1dAVb",
            "RBBB",
            "LBBB",
        ]
    )
    rhythm_classes: list = field(
        default_factory = lambda: [
            "SB",
            "AF",
            "ST",
        ]
    )
    use_superclasses: bool = True
    with_test: bool = True
    data_frac: float = 1.0

import torch

from dataclasses import dataclass, field
from typing import Any

@dataclass
class Downstream_cnn_args: 
    n_classes: int = 3
    # n_classes: int = 6
    num_blocks: int = 4
    channels: list = field(default_factory = lambda: [64, 128, 192, 256])
    dropout_rate: float = 0.2

@dataclass
class LoadDataConfig:
    hdf5_path: str = '/home/josegfer/code/code15/code15.h5'
    metadata_path: str = '/home/josegfer/code/code15/exams.csv'

    batch_size: int = 2

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

    test_size: float = 2 / 8
    val_size: float = 2 / 8

    random_seed: int = 0
    data_dtype: Any = torch.float32
    output_dtype: Any = torch.long

    use_fake_data: bool = True
    fake_h5_path: str = '/home/josegfer/code/codefake/example_exams.h5'
    fake_csv_path: str = '/home/josegfer/code/codefake/exams.csv'

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
    # use_superclasses: bool = False
    
    with_test: bool = True
    data_frac: float = 1.0
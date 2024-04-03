
import h5py
import pandas as pd
import torch
import numpy as np

class MyDataloader:
    def __init__(
        self,
        h5_data,
        h5_ids,
        h5_indices,
        metadata,
        exam_id_col,
        output_cols,
        batch_size,
        data_dtype,
        output_dtype,
        shuffle = True,
    ):
        self.h5_data = h5_data
        self.h5_ids = h5_ids
        self.h5_indices = h5_indices

        self.metadata = metadata
        self.exam_id_col = exam_id_col
        self.output_cols = output_cols

        self.batch_size = batch_size

        self.data_dtype = data_dtype
        self.output_dtype = output_dtype

        self.with_output = self.output_cols is not None

        self.dataset_size = len(h5_indices)
        self.num_batches = int(np.ceil(self.dataset_size / self.batch_size))
        self.current_batch = 0
        self.shuffle = shuffle

    def __iter__(self):
        self.current_batch = 0
        if self.shuffle:
            np.random.shuffle(self.h5_indices)
        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        # Calculate start and end indices for the current batch
        start_idx = self.current_batch * self.batch_size
        end_idx = min((self.current_batch + 1) * self.batch_size, self.dataset_size)

        # Increment the batch counter
        self.current_batch += 1

        # Get the shuffled indices for the current batch
        batch_indices = self.h5_indices[start_idx:end_idx]

        # Sort the indices for h5py compatibility
        sorted_indices = np.sort(batch_indices)

        # Fetch the data using sorted indices
        sorted_data = self.h5_data[sorted_indices]

        # Convert the reordered data to a tensor
        batch_tensor = [torch.tensor(sorted_data, dtype=self.data_dtype)]

        # Reorder h5_ids to match the shuffled order
        h5_ids = self.h5_ids[sorted_indices]
        batch_tensor.append(h5_ids)

        if self.with_output:
            output = []
            for id in h5_ids:
                output_list = list(
                    self.metadata.loc[self.metadata[self.exam_id_col] == id][
                        self.output_cols
                    ].values[0]
                )
                output.append(output_list)
            batch_tensor.append(torch.tensor(output, dtype=self.output_dtype))

        return batch_tensor

class LoadData:
    def __init__(
        self,
        hdf5_path,
        metadata_path,
        batch_size,
        exam_id_col,
        patient_id_col,
        tracing_col,
        output_col,
        tracing_dataset_name,
        exam_id_dataset_name,
        val_size,
        test_size,
        random_seed,
        data_dtype,
        output_dtype,
        use_fake_data,
        fake_h5_path,
        fake_csv_path,
        use_superclasses,
        block_classes,
        rhythm_classes,
        with_test,
        data_frac,
    ):
        if use_fake_data:
            hdf5_path = fake_h5_path
            metadata_path = fake_csv_path
            print("--> WARNING: USING RANDOM FAKE DATA!!")
        
        self.hdf5_file = h5py.File(hdf5_path, "r")
        self.hdf5_path = hdf5_path

        self.batch_size = batch_size
        self.current_iter = 0

        self.exam_id_col = exam_id_col
        self.patient_id_col = patient_id_col
        self.tracing_col = tracing_col
        self.output_col = output_col
        self.with_output = output_col is not None

        self.tracing_dataset_name = tracing_dataset_name
        self.exam_id_dataset_name = exam_id_dataset_name

        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(
            self.metadata_path,
        )

        self.exam_id_dataset = self.hdf5_file[self.exam_id_dataset_name]
        self.tracing_dataset = self.hdf5_file[self.tracing_dataset_name]

        self.val_size = val_size
        self.test_size = test_size

        self.random_seed = random_seed

        self.data_dtype = data_dtype
        self.output_dtype = output_dtype

        self.with_test = with_test
        self.data_frac = data_frac

        self.dataset_size = self.metadata.shape[0]

        self.train_set_size = int(
            self.dataset_size
            * (1 - self.val_size - (self.test_size if self.with_test else 0))
        )
        self.val_set_size = int(self.dataset_size * self.val_size)
        self.test_set_size = (
            int(self.dataset_size * self.test_size) if self.with_test else 0
        )

        self.train_metadata = None
        self.val_metadata = None
        self.test_metadata = None

        self.use_superclasses = use_superclasses
        self.block_classes = block_classes
        self.rhythm_classes = rhythm_classes

        self.split_metadata()

    def get_train_set_size(self):
        return self.train_set_size

    def get_val_set_size(self):
        return self.val_set_size

    def get_test_set_size(self):
        return self.test_set_size

    def split_metadata(self):
        self.metadata["block_class"] = self.metadata[self.block_classes].any(axis=1)
        self.metadata["rhythm_class"] = self.metadata[self.rhythm_classes].any(axis=1)
        self.metadata["normal_class"] = ~self.metadata[self.output_col].any(axis=1)

        if self.use_superclasses:
            self.output_col = ["block_class", "rhythm_class", "normal_class"]

        patient_ids = self.metadata[self.patient_id_col].unique()

        np.random.seed(self.random_seed)
        np.random.shuffle(patient_ids)

        num_train = int(len(patient_ids) * (1 - self.test_size - self.val_size))
        num_val = int(len(patient_ids) * self.val_size)

        self.train_ids = set(patient_ids[:num_train])
        self.val_ids = set(patient_ids[num_train : num_train + num_val])

        self.train_metadata = self.metadata.loc[
            self.metadata[self.patient_id_col].isin(self.train_ids)
        ].reset_index(drop=True)

        self.val_metadata = self.metadata.loc[
            self.metadata[self.patient_id_col].isin(self.val_ids)
        ].reset_index(drop=True)

        if self.with_test:
            self.test_ids = set(patient_ids[num_train + num_val :])

            self.test_metadata = self.metadata.loc[
                self.metadata[self.patient_id_col].isin(self.test_ids)
            ].reset_index(drop=True)

        self.check_dataleakage()

    def get_dataloader(self, metadata, partial=False):
        if partial:
            metadata = metadata.sample(
                frac=self.data_frac,
                replace=False,
                random_state=self.random_seed,
                ignore_index=True,
            )

        data_indices = np.where(
            np.isin(
                self.exam_id_dataset[:],
                metadata[self.exam_id_col],
            )
        )[0]

        return MyDataloader(
            h5_data=self.tracing_dataset,
            h5_ids=self.exam_id_dataset,
            h5_indices=data_indices,
            metadata=self.metadata,
            exam_id_col=self.exam_id_col,
            output_cols=self.output_col,
            batch_size=self.batch_size,
            data_dtype=self.data_dtype,
            output_dtype=self.output_dtype,
        )

    def check_dataleakage(self):
        train_ids = set(self.train_metadata[self.exam_id_col].unique())
        val_ids = set(self.val_metadata[self.exam_id_col].unique())

        # Check for intersection between any two sets of IDs
        assert (
            len(train_ids.intersection(val_ids)) == 0
        ), "Some IDs are present in both train and validation sets."

        if self.with_test:
            test_ids = set(self.test_metadata[self.exam_id_col].unique())

            assert (
                len(train_ids.intersection(test_ids)) == 0
            ), "Some IDs are present in both train and test sets."
            assert (
                len(val_ids.intersection(test_ids)) == 0
            ), "Some IDs are present in both validation and test sets."

    def get_train_dataloader(self, partial=False):
        return self.get_dataloader(self.train_metadata, partial)

    def get_val_dataloader(self, partial=False):
        return self.get_dataloader(self.val_metadata, partial)

    def get_test_dataloader(self):
        return self.get_dataloader(self.test_metadata) if self.with_test else None

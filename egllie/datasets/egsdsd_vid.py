import numpy as np
import os
from torch.utils.data import Dataset, ConcatDataset
import cv2
import random
import torch
from egllie.datasets.egsdsd import egsdsd_withNE_dataset
from egllie.datasets.utils import ConcatDatasetCustom


class SequenceSDSD(Dataset):
    """Load time-synchronized sequence data of SDSD dataset"""

    def __init__(self, dataset_root, center_cropped_height, random_cropped_width, seq, is_train, voxel_grid_channel, is_split_event, is_indoor,
                 sequence_length=16, step_size=16):
        assert sequence_length > 0
        assert step_size > 0

        self.L = sequence_length

        self.dataset = egsdsd_withNE_dataset(
            dataset_root,
            center_cropped_height,
            random_cropped_width,
            seq,
            is_train,
            voxel_grid_channel,
            is_split_event,
            is_indoor
        )

        self.step_size = step_size
        if self.L >= len(self.dataset):
            self.length = 0
        else:
            self.length = (len(self.dataset) - self.L) // self.step_size + 1

        print(f"{seq} sequence dataset length: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        """Return list of sequences containing synchronized event-image pairs"""
        assert i >= 0
        assert i < self.length

        # Generate random seed, pass to transform function of each item in the sequence
        # Ensure all items in the sequence are transformed in the same way
        seed = random.randint(0, 2**32)

        sequence = []

        # Add first element
        k = 0
        j = i * self.step_size
        item = self.dataset.getitem_with_seed(j, seed)
        sequence.append(item)

        # Add remaining sequence elements
        for n in range(self.L - 1):
            k += 1
            item = self.dataset.getitem_with_seed(j + k, seed)
            sequence.append(item)

        return sequence


def get_egsdsd_withNE_dataset_vid(
    dataset_root,
    center_cropped_height,
    random_cropped_width,
    is_train,
    is_split_event,
    voxel_grid_channel,
    is_indoor,
    sequence_length=16,
    step_size=16,
    dataset_flag=False
):
    """Build SDSD video sequence dataset

    Args:
        dataset_flag: If True, return ConcatDatasetCustom to track video boundaries during testing
    """
    all_seqs = os.listdir(dataset_root)
    all_seqs.sort()

    seq_dataset_list = []

    for seq in all_seqs:
        if os.path.isdir(os.path.join(dataset_root, seq)):
            # Load each sequence individually
            seq_dataset_list.append(
                SequenceSDSD(
                    dataset_root,
                    center_cropped_height,
                    random_cropped_width,
                    seq,
                    is_train,
                    voxel_grid_channel,
                    is_split_event,
                    is_indoor,
                    sequence_length=sequence_length,
                    step_size=step_size
                )
            )

    # Merge all sequence datasets
    # When dataset_flag=True, return ConcatDatasetCustom to track video boundaries during testing
    if dataset_flag:
        all_seq_dataset = ConcatDatasetCustom(seq_dataset_list)
    else:
        all_seq_dataset = ConcatDataset(seq_dataset_list)

    return all_seq_dataset

import bisect
from torch.utils.data import ConcatDataset


class ConcatDatasetCustom(ConcatDataset):
    """Custom ConcatDataset that also returns the dataset index.

    This is used for video validation to track which video sequence
    a frame belongs to, enabling proper RNN state reset at video boundaries.
    """

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx

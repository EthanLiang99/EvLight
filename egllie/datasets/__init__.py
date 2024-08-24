from egllie.datasets.eglol import get_eglol_withNE_dataset
from egllie.datasets.egsdsd import get_egsdsd_withNE_dataset
from os.path import join


def get_dataset(config):
    if config.NAME == "get_eglol_withNE_dataset":
        return (
            get_eglol_withNE_dataset(
                dataset_root=join(config.root, "train"),
                center_cropped_height=config.img_height,
                random_cropped_width=config.img_width,
                is_train=True,
                is_split_event=config.is_split_event,
                voxel_grid_channel=config.voxel_grid_channel
            ),
            get_eglol_withNE_dataset(
                dataset_root=join(config.root, "test"),
                center_cropped_height=config.img_height,
                random_cropped_width=config.img_width,
                is_train=False,
                is_split_event=config.is_split_event,
                voxel_grid_channel=config.voxel_grid_channel
            ),
        )
    elif config.NAME == "get_egsdsd_withNE_dataset":
        return (
            get_egsdsd_withNE_dataset(
                dataset_root=join(config.root, "train"),
                center_cropped_height=config.img_height,
                random_cropped_width=config.img_width,
                is_train=True,
                is_split_event=config.is_split_event,
                voxel_grid_channel=config.voxel_grid_channel,
                is_indoor=True
            ),
            get_egsdsd_withNE_dataset(
                dataset_root=join(config.root, "test"),
                center_cropped_height=config.img_height,
                random_cropped_width=config.img_width,
                is_train=False,
                is_split_event=config.is_split_event,
                voxel_grid_channel=config.voxel_grid_channel,
                is_indoor=True
            ),
        )
    else:
        raise ValueError(f"Unknown dataset: {config.NAME}")

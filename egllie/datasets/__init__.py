from egllie.datasets.eglol import get_eglol_withNE_dataset
from egllie.datasets.egsdsd import get_egsdsd_withNE_dataset
from egllie.datasets.eglol_vid import get_eglol_withNE_dataset_vid
from egllie.datasets.egsdsd_vid import get_egsdsd_withNE_dataset_vid
from os.path import join


def get_dataset(config):
    # Image dataset
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
    # Video dataset
    elif config.NAME == "get_eglol_withNE_dataset_vid":
        return (
            get_eglol_withNE_dataset_vid(
                dataset_root=join(config.root, "train"),
                center_cropped_height=config.img_height,
                random_cropped_width=config.img_width,
                is_train=True,
                is_split_event=config.is_split_event,
                voxel_grid_channel=config.voxel_grid_channel,
                sequence_length=config.get('sequence_length', 16),
                step_size=8,  # Align with egllie-release-vid, use 50% overlap window
                dataset_flag=False
            ),
            get_eglol_withNE_dataset_vid(
                dataset_root=join(config.root, "test"),
                center_cropped_height=config.img_height,
                random_cropped_width=config.img_width,
                is_train=False,
                is_split_event=config.is_split_event,
                voxel_grid_channel=config.voxel_grid_channel,
                sequence_length=1,  # Return single frame during testing
                step_size=1,  # Evaluate all frames
                dataset_flag=True  # Return ConcatDatasetCustom to track video boundaries
            ),
        )
    elif config.NAME == "get_egsdsd_withNE_dataset_vid":
        return (
            get_egsdsd_withNE_dataset_vid(
                dataset_root=join(config.root, "train"),
                center_cropped_height=config.img_height,
                random_cropped_width=config.img_width,
                is_train=True,
                is_split_event=config.is_split_event,
                voxel_grid_channel=config.voxel_grid_channel,
                is_indoor=True,
                sequence_length=config.get('sequence_length', 16),
                step_size=8,  # Align with egllie-release-vid, use 50% overlap window
                dataset_flag=False
            ),
            get_egsdsd_withNE_dataset_vid(
                dataset_root=join(config.root, "test"),
                center_cropped_height=config.img_height,
                random_cropped_width=config.img_width,
                is_train=False,
                is_split_event=config.is_split_event,
                voxel_grid_channel=config.voxel_grid_channel,
                is_indoor=True,
                sequence_length=1,  # Return single frame during testing
                step_size=1,  # Evaluate all frames
                dataset_flag=True  # Return ConcatDatasetCustom to track video boundaries
            ),
        )
    else:
        raise ValueError(f"Unknown dataset: {config.NAME}")

import numpy as np
import os
from torch.utils.data import Dataset, ConcatDataset
import cv2
import random
import torch
from absl.logging import info

class egsdsd_withNE_dataset(Dataset):
    def __init__(
        self,
        dataset_root,
        height,
        width,
        seq_name,
        is_train,
        voxel_grid_channel,
        is_split_event,
        is_indoor
    ):
        self.H = height
        self.W = width
        self.noraml_img_folder = os.path.join(dataset_root, seq_name, "normal")
        self.noraml_img_list_all = os.listdir(self.noraml_img_folder)  #remove event file
        # skipping the first three and last three split event files since they contains few event info
        self.noraml_img_list = sorted(list(filter(lambda x: 'png' in x, self.noraml_img_list_all)))[3:-2] 
        self.is_split_event = is_split_event
        self.is_indoor = is_indoor

        
        self.low_img_folder = os.path.join(dataset_root, seq_name, "low")
        self.low_img_list_all = os.listdir(self.low_img_folder)  #remove event file
        # skipping the first three and last three split event files since they contains few event info
        self.low_img_list = sorted(list(filter(lambda x: 'png' in x, self.low_img_list_all)))[3:-2]


        if self.is_split_event:
            self.noraml_ev_list = sorted(list(filter(lambda x: 'npz' in x, self.noraml_img_list_all)))[3:-2]
            self.low_ev_list = sorted(list(filter(lambda x: 'npz' in x, self.low_img_list_all)))[3:-2]
        else:
            self.normal_event_file = os.path.join(
            dataset_root, seq_name, "normal", "normalight_event.npz"
        )
            self.low_event_file = os.path.join(
            dataset_root, seq_name, "low", "lowlight_event.npz"
        )
            
                 
        self.num_input = len(self.low_img_list)
        self.ev_idx = None
        self.events = None
        self.center_cropped_height = 256
        self.random_cropped_width = 256
        self.is_train = is_train
        self.seq_name = seq_name
        self.voxel_grid_channel = voxel_grid_channel

    def __len__(self):
        return self.num_input

    def get_event(self, idx):
        """Split single event from the whole event file """
        # if there is no previous frame, set start_t as the timestamp of first event
        if idx == 0:
            start_t = self.events[0, 0]
        else:
            start_t = int(self.low_img_list[idx - 1][:-4])
        
        # if there is no next frame, set end_t as the timestamp of last event
        if idx == self.num_input - 1:
            end_t = self.events[-1, 0]
        else:
            end_t = int(self.low_img_list[idx + 1][:-4])

        ev_start_idx_list = np.where(self.events[:, 0] > start_t)
        ev_start_idx = ev_start_idx_list[0][0]
        ev_end_idx_list = np.where(self.events[:, 0] < end_t)
        ev_end_idx = ev_end_idx_list[0][-1]
        event = self.events[ev_start_idx:ev_end_idx]

        return event

    def _crop(self, input_frame_list, events_list):
        """crop frame and events
           training: random crop
           testing: central crop 
        """
        if self.is_train:
            min_y = random.randint(0, self.W - self.random_cropped_width)
            min_x = random.randint(0, self.H - self.center_cropped_height)
        else:
            min_y = (self.W - self.random_cropped_width) // 2
            min_x = (self.H - self.center_cropped_height) //2
        max_y = min_y + self.random_cropped_width
        max_x = min_x + self.center_cropped_height
        crop_image_list = []
        for input_frame in input_frame_list:
            input_frames = input_frame[min_x:max_x, min_y:max_y, :]
            input_frames_torch = torch.from_numpy(input_frames).permute(2,0,1).float() 
            crop_image_list.append(input_frames_torch)
    
        output_events_list = []
        for events in events_list:
            mask_x = torch.where(
                (events[:, 2] < max_x) & (events[:, 2] >= min_x)
            )
            event_x = torch.index_select(events, 0, mask_x[0])
            mask_y = torch.where(
                (event_x[:, 1] < max_y) & (event_x[:, 1] >= min_y)
            )
            event_y = torch.index_select(event_x, 0, mask_y[0])
            event = event_y.clone()
            event[:, 2] = event_y[:, 2] - min_x
            event[:, 1] = event_y[:, 1] - min_y
            output_events_list.append(event)
        return crop_image_list, output_events_list

    def _illumiantion_map(self, img):
        """obtain illumination prior"""
        intial_illumiantion_map = np.max(img, axis=2)
        return intial_illumiantion_map

    def _generate_voxel_grid(self, event):
        """obtain voxel grid """
        width, height = self.random_cropped_width, self.center_cropped_height
        event_start = event[0, 0]
        event_end = event[-1, 0]
        
        ch = (
            event[:, 0].to(torch.float32)
            / (event_end - event_start)
            * self.voxel_grid_channel
        ).long()
        torch.clamp_(ch, 0, self.voxel_grid_channel - 1)
        ex = event[:, 1].long()
        ey = event[:, 2].long()
        ep = event[:, 3].to(torch.float32)
        ep[ep == 0] = -1

        voxel_grid = torch.zeros(
            (self.voxel_grid_channel, height, width), dtype=torch.float32
        )
        voxel_grid.index_put_((ch, ey, ex), ep, accumulate=True)

        return voxel_grid
    

    def __getitem__(self, index):
        # 1. load event
        if (self.events is None) & (self.is_split_event == False) :
            # load event from the whole event file
            if self.is_indoor:
                events = np.load(self.low_event_file)
            else:
                events = np.load(self.normal_event_file)
        elif self.is_split_event == True:
            # load event from the split event file
            try:
                if self.is_indoor:
                    events = np.load(os.path.join(self.low_img_folder, self.low_ev_list[index]))
                else:
                    events = np.load(os.path.join(self.noraml_img_folder, self.noraml_ev_list[index]))
            except:
                info(f"loading event error @ seq: {self.seq_name}")
                info(f"loading event error @ index: {index}")
                info(self.low_ev_list)
        else:
            raise ValueError('w/o assign event')

        self.events = events["arr_0"]
        self.events = self.events.astype(np.int32)
        # detect loading problem
        if (not self.events.size) or (self.events.shape[0]==0):
            if self.is_indoor:
                events = np.load(os.path.join(self.low_img_folder, self.low_ev_list[index]))
            else:
                events = np.load(os.path.join(self.noraml_img_folder, self.noraml_ev_list[index]))
            self.events = events["arr_0"]
            self.events = self.events.astype(np.int32)
            info(f"loading event error @ seq: {self.seq_name}")
            info(f"loading event error @ index: {index}")
        self.events_normal = self.events


        if self.is_split_event == False:
            try:
                event_input = self.get_event(index)
            except:
                print(f"loading event error @ seq: {self.seq_name}")
        else:
            event_input = self.events
        
        del events

        # 2. load image & obtain illumination map prior
        img_low = cv2.cvtColor(
            cv2.imread(
                os.path.join(self.low_img_folder, self.low_img_list[index])
            ),
            cv2.COLOR_BGR2RGB,
        )
        img_blur = cv2.blur(img_low,(5,5))
        img_low_illumination_map = self._illumiantion_map(img_low)
        img_gt = cv2.cvtColor(
            cv2.imread(
                os.path.join(
                    self.noraml_img_folder, self.noraml_img_list[index]
                )
            ),
            cv2.COLOR_BGR2RGB,
        )
        # h, w -> h, w, c
        img_low_illumination_map = np.expand_dims(
            img_low_illumination_map / 255, axis=-1
        )


        event_input_torch = torch.from_numpy(event_input)


        crop_img_list, crop_event_list = self._crop(
            [
                img_low,
                img_gt,
                img_low_illumination_map,
                img_blur
            ],
            [event_input_torch],
        )

        input_voxel_grid_list = []
        # obtain event voxel grid
        for crop_event in crop_event_list:
            crop_event[:, 0] = crop_event[:, 0] - crop_event[0, 0]
            input_voxel_grid = self._generate_voxel_grid(crop_event)
            input_voxel_grid_list.append(input_voxel_grid)
        
        del (
            event_input,
            event_input_torch,
            crop_event_list,
        )

        sample = {
            "lowligt_image": crop_img_list[0]/255,
            "normalligt_image": crop_img_list[1]/255,
            "event_free": input_voxel_grid_list[0],
            "lowlight_image_blur": crop_img_list[3]/255,
            "ill_list": [
                crop_img_list[2],
            ],
            "seq_name": self.seq_name,
            "frame_id": self.low_img_list[index].split(".")[0],
        }

        # reduce memory cost
        self.events = None

        return sample

    def getitem_with_seed(self, index, seed):
        """For video sequence data loading, using specified random seed to maintain data augmentation consistency across sequence frames"""
        # 1. load event
        if (self.events is None) & (self.is_split_event == False):
            # load event from the whole event file
            if self.is_indoor:
                events = np.load(self.low_event_file)
            else:
                events = np.load(self.normal_event_file)
        elif self.is_split_event == True:
            # load event from the split event file
            try:
                if self.is_indoor:
                    events = np.load(os.path.join(self.low_img_folder, self.low_ev_list[index]))
                else:
                    events = np.load(os.path.join(self.noraml_img_folder, self.noraml_ev_list[index]))
            except:
                info(f"loading event error @ seq: {self.seq_name}")
                info(f"loading event error @ index: {index}")
                info(self.low_ev_list)
        else:
            raise ValueError('w/o assign event')

        self.events = events["arr_0"]
        self.events = self.events.astype(np.int32)
        # detect loading problem
        if (not self.events.size) or (self.events.shape[0] == 0):
            if self.is_indoor:
                events = np.load(os.path.join(self.low_img_folder, self.low_ev_list[index]))
            else:
                events = np.load(os.path.join(self.noraml_img_folder, self.noraml_ev_list[index]))
            self.events = events["arr_0"]
            self.events = self.events.astype(np.int32)
            info(f"loading event error @ seq: {self.seq_name}")
            info(f"loading event error @ index: {index}")
        self.events_normal = self.events

        if self.is_split_event == False:
            try:
                event_input = self.get_event(index)
            except:
                print(f"loading event error @ seq: {self.seq_name}")
        else:
            event_input = self.events

        del events

        # 2. load image & obtain illumination map prior
        img_low = cv2.cvtColor(
            cv2.imread(
                os.path.join(self.low_img_folder, self.low_img_list[index])
            ),
            cv2.COLOR_BGR2RGB,
        )
        img_blur = cv2.blur(img_low, (5, 5))
        img_low_illumination_map = self._illumiantion_map(img_low)
        img_gt = cv2.cvtColor(
            cv2.imread(
                os.path.join(
                    self.noraml_img_folder, self.noraml_img_list[index]
                )
            ),
            cv2.COLOR_BGR2RGB,
        )
        # h, w -> h, w, c
        img_low_illumination_map = np.expand_dims(
            img_low_illumination_map / 255, axis=-1
        )

        event_input_torch = torch.from_numpy(event_input)

        # Set random seed to maintain sequence consistency
        random.seed(seed)
        crop_img_list, crop_event_list = self._crop(
            [
                img_low,
                img_gt,
                img_low_illumination_map,
                img_blur
            ],
            [event_input_torch],
        )

        input_voxel_grid_list = []
        # obtain event voxel grid
        for crop_event in crop_event_list:
            crop_event[:, 0] = crop_event[:, 0] - crop_event[0, 0]
            input_voxel_grid = self._generate_voxel_grid(crop_event)
            input_voxel_grid_list.append(input_voxel_grid)

        del (
            event_input,
            event_input_torch,
            crop_event_list,
        )

        sample = {
            "lowligt_image": crop_img_list[0] / 255,
            "normalligt_image": crop_img_list[1] / 255,
            "event_free": input_voxel_grid_list[0],
            "lowlight_image_blur": crop_img_list[3] / 255,
            "ill_list": [
                crop_img_list[2],
            ],
            "seq_name": self.seq_name,
            "frame_id": self.low_img_list[index].split(".")[0],
        }

        # reduce memory cost
        self.events = None

        return sample


def get_egsdsd_withNE_dataset(
    dataset_root,
    center_cropped_height,
    random_cropped_width,
    is_train,
    is_split_event,
    voxel_grid_channel,
    is_indoor
):
    all_seqs = os.listdir(dataset_root)
    all_seqs.sort()

    seq_dataset_list = []

    for seq in all_seqs:
        if os.path.isdir(os.path.join(dataset_root, seq)):
            seq_dataset_list.append(
                egsdsd_withNE_dataset(
                    dataset_root,
                    center_cropped_height,
                    random_cropped_width,
                    seq,
                    is_train,
                    voxel_grid_channel,
                    is_split_event,
                    is_indoor
                )
            )
    return ConcatDataset(seq_dataset_list)

    


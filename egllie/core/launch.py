#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import random
import shutil
import time
from collections import OrderedDict
from os.path import isfile, join

import cv2
import numpy as np
import torch
import torch.nn as nn
from absl.logging import debug, flags, info
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from egllie.core.optimizer import Optimizer
from egllie.datasets import get_dataset
from egllie.losses import get_loss,AverageMeter,get_metric
from egllie.models import get_model


FLAGS = flags.FLAGS

def rot_aug (batch):
    """This function is used for augmentation with random rotation """
    rot_times =  np.random.randint(0,4)
    batch['lowligt_image'] = torch.rot90(batch['lowligt_image'],k=rot_times,dims=[2,3])
    batch['normalligt_image'] = torch.rot90(batch['normalligt_image'],k=rot_times,dims=[2,3])
    batch['event_free'] = torch.rot90(batch['event_free'],k=rot_times,dims=[2,3])
    batch['lowlight_image_blur'] = torch.rot90(batch['lowlight_image_blur'],k=rot_times,dims=[2,3])
    batch['ill_list'] = [torch.rot90(batch['ill_list'][i],k=rot_times,dims=[2,3]) for i in range(len(batch['ill_list']))]

    return batch


def move_tensors_to_cuda(dictionary_of_tensors):
    if isinstance(dictionary_of_tensors, dict):
        return {
            key: move_tensors_to_cuda(value)
            for key, value in dictionary_of_tensors.items()
        }
    if isinstance(dictionary_of_tensors, torch.Tensor):
        return dictionary_of_tensors.cuda(non_blocking=True)
    else:
        return dictionary_of_tensors

class Visualization:
    def __init__(self, visualization_config):
        """The visualization class for tesis.

        Args:
            visualization_config (EasyDict): The visualization config of testing.
        """
        self.saving_folder = join(FLAGS.log_dir, visualization_config.folder)
        os.makedirs(self.saving_folder, exist_ok=True)
        self.count = 0
        #
        self.tag = visualization_config.tag
        self.intermediate_visualization = (
            visualization_config.intermediate_visualization
        )
        info("Init Visualization:")
        info(f"  saving_folder: {self.saving_folder}")

    def visualize(self, inputs,outputs):
        def _save(image, path,normalize=False):
            if not isinstance(image, torch.Tensor):
                debug(
                    f"Image is not a tensor, but a {type(image)}, now image saved in {path}"
                )
                return
            image = image.detach()
            image = image.permute(1, 2, 0).cpu().numpy()
            if normalize:
                image_single = np.squeeze(image,axis=-1)
                image_max = np.max(image_single)
                image = image*1.0 / (image_max + 0.0001)
            image = image.clip(0, 1)
            image = (image * 255).astype(np.uint8)
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, image)
        

        info(f"video_name: {inputs['seq_name']}")
        info(f"frame_id: {inputs['frame_id']}")
        B = len(inputs["seq_name"])
        for b in range(B):
            video_name = inputs["seq_name"][b]
            frame_name = inputs["frame_id"][b]
            testfolder = join(self.saving_folder, video_name)
            os.makedirs(testfolder, exist_ok=True)
            # save output
            
            _save(
                outputs["gt"][b],
                join(testfolder, f"{frame_name}_gt.png"),
            )
            _save(
                outputs["pred"][b],
                join(testfolder, f"{frame_name}_pred.png"),
            )
        del inputs,outputs

class ParallelLaunch:
    def __init__(self, config):
        """The main class for parallel training. The entry point is the `run` method.

        Args:
            config (EasyDict): The config of an training experiment.
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6543"
        info(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        info(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
        # 0. config
        self.config = config
        # 1.1 init global random seed
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        random.seed(config.SEED)
        np.random.seed(config.SEED)
        # 1.2 init the tensorboard log dir
        self.tb_recoder = SummaryWriter(FLAGS.log_dir)
        # 2. VISUALIZE setting
        self.visualizer = None
        if config.VISUALIZE:
            self.visualizer = Visualization(config.VISUALIZATION)

    def run(self):
        # 0. Init
        train_dataset, val_dataset = get_dataset(self.config.DATASET)
        model = get_model(self.config.MODEL)
        criterion = get_loss(self.config.LOSS)
        metrics = get_metric(self.config.METRICS)
        opt = Optimizer(self.config.OPTIMIZER, model)
        # 1. Build model
        if self.config.IS_CUDA:
            model = nn.DataParallel(model)
            model = model.cuda()
        
        if self.config.RESUME.PATH:
            if not isfile(self.config.RESUME.PATH):
                raise ValueError(f"File not found, {self.config.RESUME.PATH}")
            if self.config.IS_CUDA:
                checkpoint = torch.load(
                    self.config.RESUME.PATH,
                    map_location=lambda storage, loc: storage.cuda(0),
                )
            else:
                checkpoint = torch.load(
                    self.config.RESUME.PATH, map_location=torch.device("cpu")
                )
                new_state_dict = OrderedDict()
                for k, v in checkpoint["state_dict"].items():
                    name = k[7:]
                    new_state_dict[name] = v
                checkpoint["state_dict"] = new_state_dict

            if self.config.RESUME.SET_EPOCH:
                self.config.START_EPOCH = checkpoint["epoch"]
                opt.optimizer.load_state_dict(checkpoint["optimizer"])
                opt.scheduler.load_state_dict(checkpoint["scheduler"])
                ### fix bug of pytorch 1.12
                opt.param_groups[0]['capturable'] = True
            model.load_state_dict(checkpoint["state_dict"],strict=True)
        # 2. Build Dataloader
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.JOBS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.JOBS,
            pin_memory=True,
            drop_last=True,
        )
        # 3. if test only
        if self.config.TEST_ONLY:
            self.valid(val_loader, model, criterion, metrics, 0)
            return
        # 4. train
        min_loss = 123456789.0
        for epoch in range(self.config.START_EPOCH, self.config.END_EPOCH):
            self.train(train_loader, model, criterion, metrics, opt, epoch)
            # save checkpoint
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": opt.optimizer.state_dict(),
                "scheduler": opt.scheduler.state_dict(),
            }
            model_dict = {
                "state_dict": model.state_dict(),
            }
            path = join(self.config.SAVE_DIR, "checkpoint.pth.tar")
            time.sleep(1)
            # valid
            if epoch % self.config.VAL_INTERVAL == 0:
                torch.save(checkpoint, path)
                val_loss = self.valid(
                    val_loader, model, criterion, metrics, epoch
                )
                if val_loss < min_loss:
                    min_loss = val_loss
                    copy_path = join(self.config.SAVE_DIR, "model_best.pth.tar")
                    shutil.copy(path, copy_path)
            # train
            if epoch % self.config.MODEL_SANING_INTERVAL == 0:
                path = join(
                    self.config.SAVE_DIR,
                    f"checkpoint-{str(epoch).zfill(3)}.pth.tar",
                )
                torch.save(checkpoint, path)
            path = join(
                    self.config.SAVE_DIR,
                    f"model-{str(epoch).zfill(3)}.pth.tar",
                )
            torch.save(model_dict, path)

    def train(self, train_loader, model, criterion, metrics, opt, epoch):
        model.train()
        info(
            f"Train Epoch[{epoch}/{self.config.END_EPOCH}]:len({len(train_loader)})"
        )
        length = len(train_loader)
        # 1. init meter
        losses_meter = {"TotalLoss": AverageMeter(f"Valid/TotalLoss")}
        for config in self.config.LOSS:
            losses_meter[config.NAME] = AverageMeter(f"Train/{config.NAME}")
        metric_meter = {}
        for config in self.config.METRICS:
            metric_meter[config.NAME] = AverageMeter(f"Train/{config.NAME}")
        batch_time_meter = AverageMeter("Train/BatchTime")
        # 2. start a training epoch
        start_time = time.time()
        time_recoder = time.time()
        scaler = torch.cuda.amp.GradScaler()
        
        
        for index, batch in enumerate(train_loader):
            if self.config.IS_CUDA:
                batch = move_tensors_to_cuda(batch)
            batch = rot_aug(batch)
            if self.config.MIX_PRECISION:
                with torch.cuda.amp.autocast():
                    outputs = model(batch)
                    outputs['epoch']=epoch
                    losses, name_to_loss = criterion(outputs)
                    # 2.1 forward
                    name_to_measure = metrics(outputs)
                scaler.scale(losses).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            else:
                outputs = model(batch)
                outputs['epoch']=epoch
                losses, name_to_loss = criterion(outputs)
                # 2.1 forward
                name_to_measure = metrics(outputs)
                # 2.2 backward
                losses.backward()
                # 2.3 update weights
                opt.step()
                opt.zero_grad()
            # 2.4 update measure
            # 2.4.1 time update
            now = time.time()
            batch_time_meter.update(now - time_recoder)
            time_recoder = now
            # 2.4.2 loss update
            losses_meter["TotalLoss"].update(losses.detach().item())
            for name, loss_item in name_to_loss:
                loss_item = loss_item.detach().item()
                losses_meter[name].update(loss_item)
            # 2.4.3 measure update
            for name, measure_item in name_to_measure:
                if isinstance(measure_item, torch.Tensor):
                    measure_item = measure_item.detach().item()
                metric_meter[name].update(measure_item)
            # 2.5 log
            if index % self.config.LOG_INTERVAL == 0:
                info(
                    f"Train Epoch[{epoch}/{self.config.END_EPOCH}, {index}/{length}]:"
                )
                for name, meter in losses_meter.items():
                    info(f"    loss:    {name}: {meter.avg}")
                for name, measure in metric_meter.items():
                    info(f"    measure: {name}: {measure.avg}")
        # 3. record a training epoch
        # 3.1 record epoch time
        epoch_time = time.time() - start_time
        batch_time = batch_time_meter.avg
        info(
            f"Train Epoch[{epoch}/{self.config.END_EPOCH}]:time:epoch({epoch_time}),batch({batch_time})"
            f"lr({opt.get_lr()})"
        )
        self.tb_recoder.add_scalar(f"Train/EpochTime", epoch_time, epoch)
        self.tb_recoder.add_scalar(f"Train/BatchTime", batch_time, epoch)
        self.tb_recoder.add_scalar(f"Train/LR", opt.get_lr(), epoch)
        for name, meter in losses_meter.items():
            info(f"    loss:    {name}: {meter.avg}")
            self.tb_recoder.add_scalar(f"Train/{name}", meter.avg, epoch)
        for name, measure in metric_meter.items():
            info(f"    measure: {name}: {measure.avg}")
            self.tb_recoder.add_scalar(f"Train/{name}", measure.avg, epoch)
        # adjust learning rate
        opt.lr_schedule()

    def valid(self, valid_loader, model, criterion, metrics, epoch):
        model.eval()
        length = len(valid_loader)
        info(
            f"Valid Epoch[{epoch}/{self.config.END_EPOCH}] starting: length({length})"
        )

        # 1. init meter
        losses_meter = {"total": AverageMeter(f"Valid/TotalLoss")}
        for config in self.config.LOSS:
            losses_meter[config.NAME] = AverageMeter(f"Valid/{config.NAME}")
        metric_meter = {}
        for config in self.config.METRICS:
            metric_meter[config.NAME] = AverageMeter(f"Valid/{config.NAME}")
              

        batch_time_meter = AverageMeter("Valid/BatchTime")
        # 2. start a validating epoch
        time_recoder = time.time()
        start_time = time_recoder
        with torch.no_grad():
            for index, batch in enumerate(valid_loader):
                if self.config.IS_CUDA:
                    batch = move_tensors_to_cuda(batch)
                if self.config.MIX_PRECISION:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch)
                        outputs['epoch']=epoch
                        losses, name_to_loss = criterion(outputs)
                        # 2.2. recorder
                        name_to_measure = metrics(outputs)
                else:
                    outputs = model(batch)
                    outputs['epoch']=epoch
                    losses, name_to_loss = criterion(outputs)
                    # 2.2. recorder
                    name_to_measure = metrics(outputs)

                # 2.3 visualization
                if self.visualizer:
                    self.visualizer.visualize(batch,outputs)
                # 2.4. update measure
                now = time.time()
                batch_time_meter.update(now - time_recoder)
                time_recoder = now
                loss = (
                    losses.detach().item()
                    if isinstance(losses, torch.Tensor)
                    else losses
                )
                losses_meter["total"].update(loss)
                for name, loss_item in name_to_loss:
                    loss_item = (
                        loss_item.detach().item()
                        if isinstance(loss_item, torch.Tensor)
                        else loss_item
                    )
                    losses_meter[name].update(loss_item)
                for name, measure_item in name_to_measure:
                    measure_item = (
                        measure_item.detach().item()
                        if isinstance(measure_item, torch.Tensor)
                        else measure_item
                    )
                    metric_meter[name].update(measure_item)

                if index % self.config.LOG_INTERVAL == 0:
                    info(
                        f"Valid Epoch[{epoch}/{self.config.END_EPOCH}, {index}/{length}]:"
                    )
                    info(f"    batch-time: {batch_time_meter.avg}")
                    for name, meter in losses_meter.items():
                        info(f"    loss:    {name}: {meter.avg}")
                    for name, measure in metric_meter.items():
                        info(f"    measure: {name}: {measure.avg}")
                del batch,outputs
                torch.cuda.empty_cache()

        # 3. record a validate epoch
        # 3.1 record epoch time
        epoch_time = time.time() - start_time
        batch_time = batch_time_meter.avg
        info(
            f"Valid Epoch[{epoch}/{self.config.END_EPOCH}]:"
            f"time:epoch({epoch_time}),batch({batch_time})"
        )
        self.tb_recoder.add_scalar(f"Valid/EpochTime", epoch_time, epoch)
        self.tb_recoder.add_scalar(f"Valid/BatchTime", batch_time, epoch)
        for name, meter in losses_meter.items():
            info(f"    loss:    {name}: {meter.avg}")
            self.tb_recoder.add_scalar(f"Valid/{name}", meter.avg, epoch)
        for name, measure in metric_meter.items():
            info(f"    measure: {name}: {measure.avg}")
            self.tb_recoder.add_scalar(f"Valid/{name}", measure.avg, epoch)
        return losses_meter["total"].avg

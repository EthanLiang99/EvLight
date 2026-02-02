#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json
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
from egllie.losses import get_loss, AverageMeter, get_metric
from egllie.models import get_model


FLAGS = flags.FLAGS


class Mixing_Augment:
    """Mixup data augmentation for video sequences."""

    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(
            torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device
        self.use_identity = use_identity
        self.augments = [self.mixup]

    def mixup(self, batch):
        lam = self.dist.rsample((1, 1)).item()
        r_index = torch.randperm(batch['lowligt_image'].size(0)).to(self.device)

        batch['lowligt_image'] = lam * batch['lowligt_image'] + (1 - lam) * batch['lowligt_image'][r_index, :]
        batch['normalligt_image'] = lam * batch['normalligt_image'] + (1 - lam) * batch['normalligt_image'][r_index, :]
        batch['event_free'] = lam * batch['event_free'] + (1 - lam) * batch['event_free'][r_index, :]
        batch['lowlight_image_blur'] = lam * batch['lowlight_image_blur'] + (1 - lam) * batch['lowlight_image_blur'][r_index, :]
        batch['ill_list'] = [
            lam * batch['ill_list'][i] + (1 - lam) * batch['ill_list'][i][r_index, :]
            for i in range(len(batch['ill_list']))
        ]
        return batch

    def __call__(self, batch):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                batch = self.augments[augment](batch)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            batch = self.augments[augment](batch)
        return batch


def move_tensors_to_cuda(dictionary_of_tensors):
    """Recursively move tensors to CUDA."""
    if isinstance(dictionary_of_tensors, dict):
        return {
            key: move_tensors_to_cuda(value)
            for key, value in dictionary_of_tensors.items()
        }
    if isinstance(dictionary_of_tensors, torch.Tensor):
        return dictionary_of_tensors.cuda(non_blocking=True)
    elif isinstance(dictionary_of_tensors, list):
        return [move_tensors_to_cuda(value) for value in dictionary_of_tensors]
    else:
        return dictionary_of_tensors


class Visualization:
    """Visualization class for video testing."""

    def __init__(self, visualization_config):
        self.saving_folder = join(FLAGS.log_dir, visualization_config.folder)
        os.makedirs(self.saving_folder, exist_ok=True)
        self.count = 0
        self.tag = visualization_config.tag
        self.intermediate_visualization = visualization_config.intermediate_visualization
        info("Init Visualization:")
        info(f"  saving_folder: {self.saving_folder}")

    def visualize(self, inputs, outputs):
        def _save(image, path, normalize=False):
            if not isinstance(image, torch.Tensor):
                debug(f"Image is not a tensor, but a {type(image)}, now image saved in {path}")
                return
            image = image.detach()
            image = image.permute(1, 2, 0).cpu().numpy()
            if normalize:
                image_single = np.squeeze(image, axis=-1)
                image_max = np.max(image_single)
                image = image * 1.0 / (image_max + 0.0001)
            image = image.clip(0, 1)
            image = (image * 255).astype(np.uint8)
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, image)

        B = len(inputs["seq_name"])
        for b in range(B):
            video_name = inputs["seq_name"][b]
            frame_name = inputs["frame_id"][b]
            testfolder = join(self.saving_folder, video_name)
            os.makedirs(testfolder, exist_ok=True)
            _save(outputs["gt"][b], join(testfolder, f"{frame_name}_gt.png"))
            _save(outputs["pred"][b], join(testfolder, f"{frame_name}_pred.png"))
        del inputs, outputs


class ParallelLaunchVid:
    """Main class for video parallel training and testing."""

    def __init__(self, config):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6543"
        info(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        info(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

        self.config = config

        # Init random seed
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        random.seed(config.SEED)
        np.random.seed(config.SEED)

        # Init tensorboard
        self.tb_recoder = SummaryWriter(FLAGS.log_dir)

        # Visualization
        self.visualizer = None
        if config.VISUALIZE:
            self.visualizer = Visualization(config.VISUALIZATION)

        # Mixing augmentation for video training
        self.mixing_augmentation = Mixing_Augment(1.2, True, 'cuda')

        # Temporal loss for video sequences
        self.temporal_loss = None
        if hasattr(config, 'TEMPORAL_LOSS') and config.TEMPORAL_LOSS:
            self.temporal_loss = get_loss(config.TEMPORAL_LOSS)

    def run(self):
        # Init dataset, model, loss, metrics, optimizer
        train_dataset, val_dataset = get_dataset(self.config.DATASET)
        model = get_model(self.config.MODEL)
        criterion = get_loss(self.config.LOSS)
        metrics = get_metric(self.config.METRICS)
        opt = Optimizer(self.config.OPTIMIZER, model)

        # Video datasets (_vid) return sequences directly, no wrapper needed

        # Build model
        if self.config.IS_CUDA:
            model = nn.DataParallel(model)
            model = model.cuda()

        # Load checkpoint
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
                    name = k[7:]  # remove 'module.' prefix
                    new_state_dict[name] = v
                checkpoint["state_dict"] = new_state_dict

            if self.config.RESUME.SET_EPOCH:
                self.config.START_EPOCH = checkpoint["epoch"]
                opt.optimizer.load_state_dict(checkpoint["optimizer"])
                opt.scheduler.load_state_dict(checkpoint["scheduler"])
                opt.param_groups[0]['capturable'] = True
            model.load_state_dict(checkpoint["state_dict"], strict=False)

        # Build dataloader
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

        # Test only mode
        if self.config.TEST_ONLY:
            self.valid(val_loader, model, criterion, metrics, 0)
            return

        # Training loop
        min_loss = float('inf')
        max_psnr = 0.0

        for epoch in range(self.config.START_EPOCH, self.config.END_EPOCH):
            self.train(train_loader, model, criterion, metrics, opt, epoch)

            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": opt.optimizer.state_dict(),
                "scheduler": opt.scheduler.state_dict(),
            }
            model_dict = {"state_dict": model.state_dict()}
            path = join(self.config.SAVE_DIR, "checkpoint.pth.tar")
            time.sleep(1)

            # Validation
            if epoch % self.config.VAL_INTERVAL == 0:
                torch.save(checkpoint, path)
                val_loss, psnr_value, psnr_star_value, ssim_value = self.valid(
                    val_loader, model, criterion, metrics, epoch
                )
                if psnr_value > max_psnr:
                    min_loss = val_loss
                    max_psnr = psnr_value
                    copy_path = join(self.config.SAVE_DIR, "model_best.pth.tar")
                    shutil.copy(path, copy_path)
                info(f'Best PSNR: {max_psnr}, PSNR_star: {psnr_star_value}, SSIM: {ssim_value}')

            # Save periodic checkpoint
            if epoch % self.config.MODEL_SANING_INTERVAL == 0:
                path = join(self.config.SAVE_DIR, f"checkpoint-{str(epoch).zfill(3)}.pth.tar")
                torch.save(checkpoint, path)

            path = join(self.config.SAVE_DIR, f"model-{str(epoch).zfill(3)}.pth.tar")
            torch.save(model_dict, path)

    def train(self, train_loader, model, criterion, metrics, opt, epoch):
        model.train()
        info(f"Train Epoch[{epoch}/{self.config.END_EPOCH}]:len({len(train_loader)})")
        length = len(train_loader)

        # Init meters
        losses_meter = {"TotalLoss": AverageMeter("Train/TotalLoss")}
        for config in self.config.LOSS:
            losses_meter[config.NAME] = AverageMeter(f"Train/{config.NAME}")
        if self.temporal_loss and hasattr(self.config, 'TEMPORAL_LOSS'):
            for config in self.config.TEMPORAL_LOSS:
                losses_meter[config.NAME] = AverageMeter(f"Train/{config.NAME}")

        metric_meter = {}
        for config in self.config.METRICS:
            metric_meter[config.NAME] = AverageMeter(f"Train/{config.NAME}")
        batch_time_meter = AverageMeter("Train/BatchTime")

        start_time = time.time()
        time_recoder = time.time()
        scaler = torch.cuda.amp.GradScaler()

        for index, sequence in enumerate(train_loader):
            if self.config.MIX_PRECISION:
                with torch.cuda.amp.autocast():
                    L = len(sequence)
                    assert L > 0

                    loss_list = []
                    output_list = []

                    # Reset RNN states for new sequence
                    model.module.reset_states()

                    for i, batch in enumerate(sequence):
                        if self.config.IS_CUDA:
                            batch = move_tensors_to_cuda(batch)
                        batch = self.mixing_augmentation(batch)

                        outputs = model(batch)
                        outputs['epoch'] = epoch
                        losses, name_to_loss = criterion(outputs)

                        output_list.append(outputs)
                        loss_list.append(losses)
                        name_to_measure = metrics(outputs)

                    # Average loss over sequence
                    losses = sum(loss_list) / L

                    # Add temporal loss
                    if self.temporal_loss:
                        temporal_loss, temporal_name_to_loss = self.temporal_loss(output_list)
                        losses = losses + temporal_loss
                        name_to_loss = name_to_loss + temporal_name_to_loss

                scaler.scale(losses).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            else:
                L = len(sequence)
                assert L > 0

                loss_list = []
                output_list = []
                model.module.reset_states()

                for i, batch in enumerate(sequence):
                    if self.config.IS_CUDA:
                        batch = move_tensors_to_cuda(batch)
                    batch = self.mixing_augmentation(batch)

                    outputs = model(batch)
                    outputs['epoch'] = epoch
                    losses, name_to_loss = criterion(outputs)

                    output_list.append(outputs)
                    loss_list.append(losses)
                    name_to_measure = metrics(outputs)

                losses = sum(loss_list) / L
                if self.temporal_loss:
                    temporal_loss, temporal_name_to_loss = self.temporal_loss(output_list)
                    losses = losses + temporal_loss
                    name_to_loss = name_to_loss + temporal_name_to_loss

                losses.backward()
                opt.step()
                opt.zero_grad()

            # Update meters
            now = time.time()
            batch_time_meter.update(now - time_recoder)
            time_recoder = now

            losses_meter["TotalLoss"].update(losses.detach().item())
            for name, loss_item in name_to_loss:
                loss_item = loss_item.detach().item()
                losses_meter[name].update(loss_item)

            for name, measure_item in name_to_measure:
                if isinstance(measure_item, torch.Tensor):
                    measure_item = measure_item.detach().item()
                metric_meter[name].update(measure_item)

            # Log
            if index % self.config.LOG_INTERVAL == 0:
                info(f"Train Epoch[{epoch}/{self.config.END_EPOCH}, {index}/{length}]:")
                for name, meter in losses_meter.items():
                    info(f"    loss:    {name}: {meter.avg}")
                for name, measure in metric_meter.items():
                    info(f"    measure: {name}: {measure.avg}")

        # Record epoch
        epoch_time = time.time() - start_time
        batch_time = batch_time_meter.avg
        info(f"Train Epoch[{epoch}/{self.config.END_EPOCH}]:time:epoch({epoch_time}),batch({batch_time})lr({opt.get_lr()})")

        self.tb_recoder.add_scalar("Train/EpochTime", epoch_time, epoch)
        self.tb_recoder.add_scalar("Train/BatchTime", batch_time, epoch)
        self.tb_recoder.add_scalar("Train/LR", opt.get_lr(), epoch)

        for name, meter in losses_meter.items():
            info(f"    loss:    {name}: {meter.avg}")
            self.tb_recoder.add_scalar(f"Train/{name}", meter.avg, epoch)
        for name, measure in metric_meter.items():
            info(f"    measure: {name}: {measure.avg}")
            self.tb_recoder.add_scalar(f"Train/{name}", measure.avg, epoch)

        opt.lr_schedule()

    def valid(self, valid_loader, model, criterion, metrics, epoch):
        """Validation method - aligned with egllie-release-vid test process

        Uses ConcatDatasetCustom which returns (batch, dataset_index) tuples,
        tracks video boundaries via dataset_index, and resets RNN state when switching videos.
        Each frame is evaluated only once, RNN state remains continuous within the same video.
        """
        model.eval()
        length = len(valid_loader)
        info(f"Valid Epoch[{epoch}/{self.config.END_EPOCH}] starting: length({length})")

        valid_results = {}

        # Init meters
        losses_meter = {"total": AverageMeter("Valid/TotalLoss")}
        for config in self.config.LOSS:
            losses_meter[config.NAME] = AverageMeter(f"Valid/{config.NAME}")

        metric_meter = {}
        for config in self.config.METRICS:
            metric_meter[config.NAME] = AverageMeter(f"Valid/{config.NAME}")

        batch_time_meter = AverageMeter("Valid/BatchTime")
        time_recoder = time.time()
        start_time = time_recoder
        prev_dataset_idx = -1

        with torch.no_grad():
            for index, (batch, dataset_index) in enumerate(valid_loader):
                # batch is a list containing single frame (sequence_length=1), extract first element
                batch = batch[0]

                # Reset RNN state when switching to a new video
                if dataset_index > prev_dataset_idx:
                    info(f"Processing dataset {dataset_index}, index {index}")
                    model.module.reset_states()

                if self.config.IS_CUDA:
                    batch = move_tensors_to_cuda(batch)

                if self.config.MIX_PRECISION:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch)
                        outputs['epoch'] = epoch
                        losses, name_to_loss = criterion(outputs)
                        name_to_measure = metrics(outputs)
                else:
                    outputs = model(batch)
                    outputs['epoch'] = epoch
                    losses, name_to_loss = criterion(outputs)
                    name_to_measure = metrics(outputs)

                # Visualization
                if self.visualizer:
                    self.visualizer.visualize(batch, outputs)

                # Update meters
                now = time.time()
                batch_time_meter.update(now - time_recoder)
                time_recoder = now

                loss = losses.detach().item() if isinstance(losses, torch.Tensor) else losses
                losses_meter["total"].update(loss)

                for name, loss_item in name_to_loss:
                    loss_item = loss_item.detach().item() if isinstance(loss_item, torch.Tensor) else loss_item
                    losses_meter[name].update(loss_item)

                valid_results[batch['seq_name'][0] + '_' + batch['frame_id'][0]] = {}
                for name, measure_item in name_to_measure:
                    measure_item = measure_item.detach().item() if isinstance(measure_item, torch.Tensor) else measure_item
                    metric_meter[name].update(measure_item)
                    valid_results[batch['seq_name'][0] + '_' + batch['frame_id'][0]][name] = measure_item

                if index % self.config.LOG_INTERVAL == 0:
                    info(f"Valid Epoch[{epoch}/{self.config.END_EPOCH}, {index}/{length}]:")
                    info(f"    batch-time: {batch_time_meter.avg}")
                    for name, meter in losses_meter.items():
                        info(f"    loss:    {name}: {meter.avg}")
                    for name, measure in metric_meter.items():
                        info(f"    measure: {name}: {measure.avg}")

                del batch, outputs
                torch.cuda.empty_cache()

                prev_dataset_idx = dataset_index

        # Save validation results
        with open(join(self.config.SAVE_DIR, "valid_results.json"), "w") as f:
            json.dump(valid_results, f)

        # Record epoch
        epoch_time = time.time() - start_time
        batch_time = batch_time_meter.avg
        info(f"Valid Epoch[{epoch}/{self.config.END_EPOCH}]:time:epoch({epoch_time}),batch({batch_time})")

        self.tb_recoder.add_scalar("Valid/EpochTime", epoch_time, epoch)
        self.tb_recoder.add_scalar("Valid/BatchTime", batch_time, epoch)

        for name, meter in losses_meter.items():
            info(f"    loss:    {name}: {meter.avg}, count: {meter.count}")
            self.tb_recoder.add_scalar(f"Valid/{name}", meter.avg, epoch)
        for name, measure in metric_meter.items():
            info(f"    measure: {name}: {measure.avg}, count: {measure.count}")
            self.tb_recoder.add_scalar(f"Valid/{name}", measure.avg, epoch)

        return (
            losses_meter["total"].avg,
            metric_meter["PSNR"].avg,
            metric_meter["PSNR_star"].avg,
            metric_meter["SSIM"].avg
        )

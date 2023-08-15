import logging
import math
from pathlib import Path
from typing import Any, List, Tuple, Optional
import time
from torch import nn
from omegaconf import DictConfig
import torch
import numpy as np
from functools import reduce
import cv2
from rich.progress import track
from torch import optim
import lightning.pytorch as pl
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from torchmetrics import Accuracy
from deepdrr.utils import image_utils
import pandas as pd
from collections import Counter
import imageio.v3 as iio
import imageio
from perphix.data import PerphixDataset, PerphixContainer
from scipy.ndimage import gaussian_filter, median_filter
from hydra.utils import instantiate

from ..plots import plot_sequence_predictions
from ..metrics import eval_metrics
from ..models.transformer import UNetTransformer
from ..utils import to_numpy
from ..utils.nn_utils import detect_landmark
from ..losses import DiceLoss2D, HeatmapLoss2D

log = logging.getLogger(__name__)


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (232, 21, 21)  # (255, 0, 0)
GREEN = (50, 168, 82)  # (0, 255, 0)
BLUE = (0, 0, 255)


class PelphixModule(pl.LightningModule):
    def __init__(
        self,
        *,
        supercategories: list[str],
        supercategory_num_classes: list[int],
        num_seg_classes: int,
        num_keypoints: int,
        unet: dict[str, Any] = dict(),
        transformer: dict[str, Any] = dict(),
        optimizer: dict[str, Any] = dict(),
        scheduler: dict[str, Any] = dict(),
        use_keypoints: bool = True,
        use_segmentations: bool = True,
        results_dir: Path = Path("."),
        sequence_counts: Optional[dict[str, np.ndarray]] = None,
        test_dataset: Optional[PerphixContainer] = None,
        filter_preds: bool = False,
    ):
        """Recognition module.

        Args:
            supercategories (list[str]): List of supercategories.
            supercategory_num_classes (list[int]): List of number of classes for each supercategory, not including the bg class.
            transformer (dict[str, Any], optional): Transformer configuration. Defaults to dict().
            optimizer (dict[str, Any], optional): Optimizer configuration. Defaults to dict().
            scheduler (dict[str, Any], optional): Scheduler configuration. Defaults to dict().
            results_dir (Path, optional): Test set results directory. Defaults to Path(".").
            test_dataset (Optional[PerphixContainer], optional): Test dataset container, for mapping predictions back to
                human-readable labels. If not provided, raw labels will be used. Defaults to None.
            sequence_counts (Optional[dict[str, np.ndarray]], optional): Dictionary of sequence counts for each supercategory.
            input_images (bool, optional): Whether the input is an image. Defaults to False.
            has_mask (bool, optional): Whether the transformer should be masked to future inputs. Defaults to False.
        """
        super().__init__()
        self.supercategories = supercategories
        self.supercategory_num_classes = list(supercategory_num_classes)
        supercategory_num_classes_with_bg = [n + 1 for n in supercategory_num_classes]
        self.num_seg_classes = num_seg_classes
        self.num_keypoints = num_keypoints
        self.model = UNetTransformer(
            num_seg_classes=num_seg_classes,
            num_keypoints=num_keypoints,
            supercategory_num_classes=supercategory_num_classes_with_bg,
            transformer=transformer,
            unet=unet,
        )
        self.use_keypoints = use_keypoints
        self.use_segmentations = use_segmentations
        self.filter_preds = filter_preds

        # For training.
        self.criteria = nn.ModuleList()
        for supercategory, num_classes in zip(supercategories, supercategory_num_classes_with_bg):
            if sequence_counts is not None and supercategory in sequence_counts:
                assert sequence_counts[supercategory].shape[0] == num_classes, sequence_counts
                total = sequence_counts[supercategory].sum()
                weight = total / (sequence_counts[supercategory] + 1e-8)
                weight[0] = 0  # Ignore bg class.
                weight = weight / weight.sum()
                weight = torch.tensor(weight, dtype=torch.float32)
                log.info(f"Using weights for {supercategory}: {weight}")
                self.criteria.append(nn.CrossEntropyLoss(weight=weight))
            else:
                self.criteria.append(nn.CrossEntropyLoss())

        self.dice_loss = DiceLoss2D(skip_bg=False)
        self.heatmap_loss = HeatmapLoss2D()

        self.optimizer = optimizer
        self.scheduler = scheduler

        # For testing.
        self.results_dir = Path(results_dir).expanduser()
        self.test_dataset = test_dataset

        for supercategory, num_classes in zip(supercategories, supercategory_num_classes_with_bg):
            setattr(
                self,
                f"{supercategory}_accuracy",
                Accuracy(task="multiclass", num_classes=num_classes),
            )

        self.test_step_results = []

    def forward(
        self,
        images: torch.Tensor,
        has_mask: bool = False,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            src (torch.Tensor): (S, N, C, H, W)
            src_key_padding_mask (Optional[torch.Tensor], optional): (N, S). Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Dictionary of outputs, including:
                * outputs: num_supercategories-(N, S, num_supercategories)

        """

        outputs = self.model(images, has_mask=has_mask, src_key_padding_mask=src_key_padding_mask)
        return outputs

    def forward_with_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        masks: torch.Tensor,
        heatmaps: torch.Tensor,
        has_mask: bool = False,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with loss calculation.

        Args:
            src (torch.Tensor): (S, N, E), if input_images is false, else (S, N, C, H, W)
            labels (torch.Tensor): (num_supercategories, S, N)
            masks (torch.Tensor): (S, N, C, H, W)
            heatmaps (torch.Tensor): (S, N, C, H, W)
            has_mask (bool, optional): Whether the transformer should be masked to future inputs. Defaults to False.
            src_key_padding_mask (Optional[torch.Tensor], optional): (N, S). Defaults to None.

        """

        S, N, C, H, W = images.shape
        outputs = self.forward(images, has_mask=has_mask, src_key_padding_mask=src_key_padding_mask)

        target_label: torch.Tensor
        losses: dict[str, torch.Tensor] = dict()
        pred_labels = outputs["labels"]
        for i, (pred_label, target_label) in enumerate(zip(pred_labels, labels)):
            # output: (S, N, C)
            # label: (S, N)
            # src_key_padding_mask: (N, S)
            # Only use the frames that are in the sequence.
            C = pred_label.shape[-1]
            mask = torch.logical_not(src_key_padding_mask).transpose(1, 0)
            pred_label = torch.masked_select(pred_label, mask.unsqueeze(-1))
            pred_label = pred_label.reshape(-1, C)
            target_label = torch.masked_select(target_label, mask)
            losses[self.supercategories[i]] = self.criteria[i](pred_label, target_label)

        # Dice loss.
        pred_masks = outputs["masks"]
        if self.use_segmentations:
            losses["dice"] = self.dice_loss(
                pred_masks.reshape(S * N, -1, H, W), masks.reshape(S * N, -1, H, W)
            )

        pred_heatmaps = outputs["heatmaps"]
        if self.use_keypoints:
            losses["heatmap"] = self.heatmap_loss(
                pred_heatmaps.reshape(S * N, -1, H, W), heatmaps.reshape(S * N, -1, H, W)
            )

        loss = torch.stack([losses[k] for k in losses]).mean()
        output_dict = dict(loss=loss, losses=losses, **outputs)
        return output_dict

    def eval_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
        batch_idx: int,
        mode: str,
    ):
        """Unified eval step for train, val, and test.

        Args:
            batch (dict[str, torch.Tensor]): tuple containing inputs, targets. Inputs:
                images (torch.Tensor): (N, S, C, H, W)

                labels (list[torch.Tensor]): (N, S, num_supercategories)
                src_key_padding_mask (Optional[torch.Tensor], optional): (N, S). Defaults to None.
            batch_idx (int): Batch index
            mode (str): "train", "val", or "test"

        """
        inputs, targets = batch
        images = inputs["images"].transpose(1, 0)  # (N, S, C, H, W) -> (S, N, C, H, W)
        src_key_padding_mask = inputs["src_key_padding_mask"]
        labels = targets["labels"].permute(
            2, 1, 0
        )  # (N, S, num_supercategories) -> (num_supercategories, S, N)
        masks = targets["masks"].transpose(1, 0)  # (N, S, C, H, W) -> (C, S, N, H, W)
        heatmaps = targets["heatmaps"].transpose(1, 0)  # (N, S, C, H, W) -> (C, S, N, H, W)

        outputs = self.forward_with_loss(
            images,
            labels,
            masks,
            heatmaps,
            has_mask=True,
            src_key_padding_mask=src_key_padding_mask,
        )
        loss = outputs["loss"]
        self.log(f"{mode}_loss/total", loss, sync_dist=True)
        for k, l in outputs["losses"].items():
            self.log(f"{mode}_loss/{k}", l, sync_dist=True)

        # Accuracy
        pred_labels = outputs["labels"]
        accs = []
        for supercategory, pred_label, label in zip(self.supercategories, pred_labels, labels):
            accuracy = getattr(self, f"{supercategory}_accuracy")
            acc = accuracy(pred_label.reshape(-1, pred_label.shape[2]), label.reshape(-1))
            self.log(f"{mode}_accuracy/{supercategory}", accuracy, sync_dist=True)
            accs.append(acc)

        acc = torch.stack(accs).mean()
        self.log(f"{mode}_accuracy/avg", acc, sync_dist=True)

        outputs["predictions"] = self._get_predictions(pred_labels)
        outputs["sorted_predictions"] = self._get_sorted_predictions(pred_labels)

        return outputs

    def _filter_logits(self, pred_logit: np.ndarray) -> np.ndarray:
        """Apply a filter to the logits.

        Args:
            pred_logits (np.ndarray): (N, S, num_classes) arrays of logits

        Returns:
            ndarray: (N, S, num_classes) array of logits
        """

        pred_logit = median_filter(pred_logit, size=(1, 3, 1), mode="nearest")
        return pred_logit

    def _get_predictions(self, pred_logits: list[torch.Tensor]) -> dict[str, np.ndarray]:
        """Get predictions from outputs.
        Args:
            pred_labels (list[torch.Tensor]): num_supercategories-list of (N, S, num_classes) arrays of logits

        Returns:
            ndarray: (N, S, num_supercategories) array of predictions as integers
        """
        predictions = []
        for i, pred_logit in enumerate(pred_logits):
            pred_logit = to_numpy(pred_logit)
            if self.filter_preds:
                pred_logit = self._filter_logits(pred_logit)

            prediction = np.argmax(pred_logit, 2).transpose(1, 0)  # (N, S)
            predictions.append(prediction)
        predictions = np.array(predictions).transpose(1, 2, 0)  # (N, S, num_supercategories)
        return predictions

    def _get_sorted_predictions(self, pred_logits: list[torch.Tensor]) -> list[np.ndarray]:
        """Get predictions from outputs.

        Args:
            pred_labels (list[torch.Tensor]): num_supercategories-list of (N, S, num_classes)

        Returns:
            ndarray: num_supercategories-list of (N, S, num_classes) array of prediction integers, sorted from most to least likely.
        """

        sorted_predictions = []
        for i, pred_logit in enumerate(pred_logits):
            pred_logit = to_numpy(pred_logit)
            if not self.training:
                pred_logit[0] = 0  # Ignore bg class
            if self.filter_preds:
                pred_logit = self._filter_logits(pred_logit)
            prediction = np.argsort(pred_logit, 2)[:, :, ::-1]
            prediction = prediction.transpose(1, 0, 2)  # (N, S, num_classes)
            sorted_predictions.append(prediction)
        return sorted_predictions

    def get_filtered_predictions(
        self,
        df: pd.DataFrame,
        sorted_preds: list[np.ndarray],
        dataset: PerphixDataset,
        image_id: int,
    ):
        """Get prediction for batch n, image s, by filtering tasks that are already done.

        Args:
            df (pd.DataFrame): History of preds
            sorted_predictions (list[np.ndarray]): num_supercategories-list of (num_classes,) array of prediction integers, sorted from most to least likely.
            dataset (PelvicWorkflowsDataset): Dataset with get_seqence_names_from_labels method
            image_id (int): Image id

        """

        # First, get the set of tasks that are already done, based on the pred_task column, for all rows with frame_number < image_id.
        # A task is "done" if a new task has been going for 3 frames
        completed = set()
        prev_task = None
        prev_task_counter = 0
        counts = Counter()
        for task in df["pred_task"]:
            counts[task] += 1
            if prev_task is None:
                prev_task = task
                continue

            if task != prev_task and counts[task] > 2 and counts[prev_task] > 2:
                completed.add(prev_task)
                prev_task = task
            elif task != prev_task and counts[task] > 2:
                # The prev_task was a blip, so clear the counts for the prev_task.
                counts[prev_task] = 0
                prev_task = task

            # if task != prev_task and counts[task] > 3 and counts[prev_task] > 3:
            #     # So if the new task has been going for 3 frames, we consider the previous task to be done.
            #     completed.add(prev_task)
            #     prev_task = task

        # Now, we can filter the predictions by only considering the first prediction that is not done.
        fingers = [0] * len(sorted_preds)
        while all([fingers[i] < len(sorted_preds[i]) - 1 for i in range(len(sorted_preds))]):
            label = np.array([pr[fingers[i]] for i, pr in enumerate(sorted_preds)])
            log.debug(f"label: {label}")
            sequence_names = dataset.get_sequence_names_from_labels(label)
            if "task" not in sequence_names:
                log.warning(f"Skipping {image_id} because no task was predicted: {sequence_names}")
            elif sequence_names["task"] not in completed:
                break
            fingers[0] += 1  # Why are we doing this?
        return sequence_names

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx):
        src = batch["src"]
        src_key_padding_mask = batch["src_key_padding_mask"]
        src = src.permute(1, 0, 2)
        outputs = self.forward(src, src_key_padding_mask=src_key_padding_mask)
        return self._get_predictions(outputs)

    def training_step(self, batch, batch_idx):
        outputs = self.eval_step(batch, batch_idx, mode="train")
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.eval_step(batch, batch_idx, mode="val")

        # Log the landmark detection accuracy
        inputs, targets = batch

        keypoints = to_numpy(targets["keypoints"])  # (N, S, num_keypoints, 2)
        heatmaps = to_numpy(targets["heatmaps"])  # (N, S, num_keypoints, H, W)

        num_present = 0
        num_detected = 0
        errors = []

        N, S, num_keypoints, H, W = heatmaps.shape
        for n in range(N):
            for s in range(S):
                for k in range(num_keypoints):
                    heatmap = heatmaps[n, s, k]
                    keypoint = keypoints[n, s, k]
                    x, y = keypoint
                    if not (0 <= x < W and 0 <= y < H):
                        continue
                    num_present += 1

                    keypoint_pred_ij = detect_landmark(heatmap)
                    if keypoint_pred_ij is None:
                        continue

                    keypoint_pred = np.array([keypoint_pred_ij[1], keypoint_pred_ij[0]])
                    error = np.linalg.norm(keypoint - keypoint_pred)
                    if error < 10:
                        num_detected += 1
                        errors.append(error)

        accuracy = num_detected / num_present
        err = np.mean(errors)
        std = np.std(errors)
        self.log(f"val_keypoint/accuracy", accuracy, sync_dist=True)
        self.log(f"val_keypoint/error", err, sync_dist=True)
        self.log(f"val_keypoint/std", std, sync_dist=True)

        return outputs["loss"]

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx):
        outputs = self.eval_step(batch, batch_idx, mode="test")
        inputs, targets = batch
        labels = to_numpy(targets["labels"])  # (N, S, num_supercategories)
        src_key_padding_mask = to_numpy(inputs["src_key_padding_mask"])  # (N, S)
        batch_size = labels.shape[0]
        seq_len = labels.shape[1]
        predictions = outputs["predictions"]
        sorted_predictions: np.ndarray = outputs["sorted_predictions"]

        # TODO: move to the base class as a classmethod/variable
        supercategory_name_mapping = {
            "task": "Corridor",
            "activity": "Activity",
            "acquisition": "View",
            "frame": "Frame",
        }

        procedure_indices = to_numpy(inputs["procedure_idx"]).astype(int)
        images: np.ndarray = (to_numpy(inputs["images"]) * 127.5 + 127.5).astype(np.uint8)
        image_ids: np.ndarray = to_numpy(inputs["image_ids"]).astype(int)
        heatmaps: np.ndarray = to_numpy(outputs["heatmaps"].transpose(1, 0))
        masks: np.ndarray = to_numpy(outputs["masks"].transpose(1, 0))

        # These values not dependent on image size
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1.2
        thickness = 2
        hspace = 160
        vspace = 25
        margin = 30

        # each maps to dataset_results_dir
        csv_paths = set()
        seq_image_dirs = set()
        mask_image_dirs = set()
        heatmap_image_dirs = set()

        for n in range(batch_size):
            procedure_idx = int(procedure_indices[n])
            dataset = self.test_dataset.get_dataset(procedure_idx)
            dataset_name = dataset.name
            dataset_results_dir = self.results_dir / dataset_name / f"{procedure_idx:09d}"
            seq_images_dir = dataset_results_dir / "images"
            seq_images_dir.mkdir(parents=True, exist_ok=True)
            seq_image_dirs.add(seq_images_dir)

            csv_path = dataset_results_dir / f"{dataset_name}_{procedure_idx:09d}.csv"
            if csv_path.exists():
                log.info(f"Appending to {csv_path}. Delete it to overwrite.")
                df = pd.read_csv(csv_path)
            else:
                df = pd.DataFrame(
                    columns=[
                        "Index",
                        "Frame Number",
                        "task",
                        "pred_task",
                        "activity",
                        "pred_activity",
                        "acquisition",
                        "pred_acquisition",
                        "frame",
                        "pred_frame",
                    ]
                )

            for s in track(
                range(seq_len),
                description=f"Processing {dataset_name} {procedure_idx:09d}",
                total=seq_len,
            ):
                if src_key_padding_mask[n, s]:
                    continue
                if image_ids[n, s] in df["Frame Number"].values:
                    # Already processed this image
                    continue

                # subtract bg class
                gt = dataset.get_sequence_names_from_labels(labels[n, s])
                # log.debug(f"{image_ids[n, s]:03d}: {labels[n,s]} -> {list(gt.values())}")

                # Filter out completed tasks
                sorted_preds = [
                    pr[n, s] for pr in sorted_predictions
                ]  # list of (num_classes,) arrays

                # Either filter the predictions or use the raw predictions
                pred = self.get_filtered_predictions(df, sorted_preds, dataset, image_ids[n, s])
                # pred = dataset.get_sequence_names_from_labels(predictions[n, s])

                labeled_image = images[n, s].transpose(1, 2, 0).copy()  # (H, W, C)

                # Undo the triplets
                labeled_image = np.stack(
                    [labeled_image[:, :, 2], labeled_image[:, :, 2], labeled_image[:, :, 2]], axis=2
                )
                log.info(f"labeled_image: {labeled_image.mean()}, {labeled_image.std()}")
                # Resize the image to 384x384
                labeled_image = cv2.resize(labeled_image, (512, 512))

                log.info(f"flipping image {image_ids[n, s]:09d} (hopefully back)")
                labeled_image = cv2.flip(labeled_image, 1)

                # Add row to df if Frame Number is not in the df
                row = pd.DataFrame(
                    {
                        "Index": [s],  # BAD
                        "Frame Number": [image_ids[n, s]],
                        **dict((k, [v]) for k, v in gt.items()),
                        **dict((f"pred_{k}", [v]) for k, v in pred.items()),
                    }
                )
                df = pd.concat([df, row], ignore_index=True)
                image_name = f"{image_ids[n, s]:09d}.png"

                # Write out the image with sequence labels
                for i, supercategory in enumerate(["task", "activity", "acquisition", "frame"]):
                    y = labeled_image.shape[0] - (vspace * i + margin)
                    supercategory_name = supercategory_name_mapping[supercategory]
                    labeled_image = cv2.putText(
                        labeled_image,
                        f"{supercategory_name}:",
                        (margin, y),
                        font,
                        font_scale,
                        BLACK,
                        thickness,
                    )
                    labeled_image = cv2.putText(
                        labeled_image,
                        f"{pred.get(supercategory, '')}",
                        (margin + hspace, y),
                        font,
                        font_scale,
                        (
                            RED
                            if pred.get(supercategory, "") != gt.get(supercategory, "None")
                            else GREEN
                        ),
                        thickness,
                    )
                    labeled_image = cv2.putText(
                        labeled_image,
                        f"({gt.get(supercategory, 'None')})",
                        (margin + 2 * hspace, y),
                        font,
                        font_scale,
                        GREEN,
                        thickness,
                    )

                image_path = seq_images_dir / f"{image_ids[n, s]:09d}.png"
                H, W = labeled_image.shape[:2]

                image = images[n, s].transpose(1, 2, 0).copy()
                image = np.stack(
                    [image[:, :, 2], image[:, :, 2], image[:, :, 2]], axis=2
                )  # last channel
                heatmap = heatmaps[n, s].transpose(1, 2, 0)
                mask = masks[n, s].transpose(1, 2, 0)

                heatmap_image = image_utils.blend_heatmaps(image, heatmap)
                heatmap_image = np.flip(heatmap_image, 1)
                heatmap_image = image_utils.as_uint8(heatmap_image)
                heatmap_image = cv2.resize(heatmap_image, (W, H))

                mask_image = image_utils.draw_masks(image, mask)
                mask_image = np.flip(mask_image, 1)
                mask_image = image_utils.as_uint8(mask_image)
                mask_image = cv2.resize(mask_image, (W, H))

                tiled_image = np.concatenate([labeled_image, heatmap_image, mask_image], axis=1)
                # tiled_image = labeled_image
                # tiled_image = cv2.cvtColor(tiled_image, cv2.COLOR_RGB2BGR)  # TODO: keep?
                image_utils.save(image_path, tiled_image)

                # TODO: make these functions scale up the image and write labels on it.
                # image_utils.save(heatmap_images_dir / image_name, heatmap_image)
                # image_utils.save(mask_images_dir / image_name, mask_image)
                log.info(f"Saved batch {batch_idx:04d} sample {n:02d}_{s:03d}")

            # Sort the df by Frame Number
            df = df.sort_values(by=["Frame Number"])
            df.to_csv(csv_path, index=False)
            csv_paths.add(csv_path)

        result = dict(
            csv_paths=csv_paths,
            seq_image_dirs=seq_image_dirs,
            mask_image_dirs=mask_image_dirs,
            heatmap_image_dirs=heatmap_image_dirs,
        )
        self.test_step_results.append(result)

    def epoch_end(self, mode: str):
        for supercategory in self.supercategories:
            accuracy = getattr(self, f"{supercategory}_accuracy")
            self.log(f"{mode}_accuracy_epoch/{supercategory}", accuracy)

    def on_train_epoch_end(self):
        self.epoch_end(mode="train")

    def on_validation_epoch_end(self):
        self.epoch_end(mode="val")

    def on_test_epoch_end(self):
        self.epoch_end(mode="test")

        csv_paths = reduce(lambda a, b: a | b, [r["csv_paths"] for r in self.test_step_results])
        for csv_path in csv_paths:
            plot_sequence_predictions(csv_path)
            eval_metrics(csv_path)

        seq_image_dirs: list[Path] = reduce(
            lambda a, b: a | b, [r["seq_image_dirs"] for r in self.test_step_results]
        )
        for seq_image_dir in seq_image_dirs:
            frames = np.array(
                [
                    cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                    for p in sorted(seq_image_dir.glob("*.png"))
                ]
            )

            dataset_results_dir = seq_image_dir.parent
            dataset_name = dataset_results_dir.parts[-2]
            procedure_idx = int(dataset_results_dir.parts[-1])

            # gif_path = dataset_results_dir / f"{dataset_name}_{procedure_idx:09d}_sequences.gif"
            # log.info(f"Writing GIF to {gif_path}...")
            # iio.imwrite(gif_path, frames, duration=500, loop=1)

            mp4_path = dataset_results_dir / f"{dataset_name}_{procedure_idx:09d}.mp4"
            log.info(f"Writing MP4 to {mp4_path}...")
            writer = imageio.get_writer(mp4_path, fps=2)
            for frame in frames:
                writer.append_data(frame)
            writer.close()

        self.test_step_results.clear()

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer, params=self.parameters())
        lr_scheduler_type = getattr(optim.lr_scheduler, self.scheduler["name"])
        lr_scheduler = lr_scheduler_type(optimizer, **self.scheduler["config"])
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler, monitor="val_accuracy/activity")

    def configure_callbacks(self):
        return [
            LearningRateMonitor(logging_interval="step"),
            DeviceStatsMonitor(),
            ModelCheckpoint(
                save_last=True,
                every_n_epochs=10,
                save_top_k=20,
                monitor="val_accuracy/activity",
                mode="max",
            ),
        ]

from __future__ import annotations
from typing import Dict, Optional, Tuple, Sequence, Iterable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from collections import defaultdict
from pathlib import Path
import json
import random
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def clip_eval_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def clip_train_transform(size: int = 224, profile: str = "standard"):
    """Build a train-only CLIP transform without changing evaluation inputs.

    ``medical_style`` applies bounded global colour, focus, and acquisition-style
    perturbations.  It deliberately avoids cross-image content mixing so that
    lesion morphology cannot leak between samples.  The pre-existing transform
    remains the default for exact backwards compatibility.
    """
    profile = str(profile).lower()
    if profile == "standard":
        return transforms.Compose([
            transforms.RandomResizedCrop(
                size,
                scale=(0.5, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    if profile == "medical_style":
        kernel_size = max(3, int(0.05 * size) // 2 * 2 + 1)
        return transforms.Compose([
            transforms.RandomResizedCrop(
                size,
                scale=(0.65, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomApply(
                [transforms.ColorJitter(0.5, 0.5, 0.5, 0.05)],
                p=0.8,
            ),
            transforms.RandomPosterize(bits=5, p=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=1.25, p=0.2),
            transforms.RandomAutocontrast(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))],
                p=0.25,
            ),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ])
    raise ValueError(
        "training augmentation profile must be standard or medical_style"
    )


class TrainCaptionDataset(Dataset):
    """Training-only dataset. Captions are represented only by cached embeddings."""

    def __init__(
        self,
        manifest_path: str | Path,
        caption_cache_path: str | Path,
        image_size: int = 224,
        augmentation_profile: str = "standard",
        minority_augmentation_profile: Optional[str] = None,
        augmentation_majority_classes: Optional[Iterable[int]] = None,
    ):
        manifest = load_manifest(manifest_path)
        self.root = Path(manifest["root"])
        self.records = manifest["splits"]["train"]
        cache = torch.load(caption_cache_path, map_location="cpu", weights_only=False)
        ids = [int(x) for x in cache["sample_ids"]]
        self.caption_by_id = {sid: cache["embeddings"][i] for i, sid in enumerate(ids)}
        expected = {int(x["sample_id"]) for x in self.records}
        if set(self.caption_by_id) != expected:
            missing = sorted(expected - set(self.caption_by_id))[:5]
            extra = sorted(set(self.caption_by_id) - expected)[:5]
            raise ValueError(f"Caption cache must equal train IDs exactly; missing={missing}, extra={extra}")
        self.labels = [int(x["label"]) for x in self.records]
        self.transform = clip_eval_transform(image_size)
        self.augmentation_profile = str(augmentation_profile).lower()
        self.augment = clip_train_transform(image_size, self.augmentation_profile)
        self.augmentation_majority_classes = frozenset(
            int(value) for value in (augmentation_majority_classes or [])
        )
        self.minority_augmentation_profile = (
            None
            if minority_augmentation_profile is None
            else str(minority_augmentation_profile).lower()
        )
        if (
            self.minority_augmentation_profile is not None
            and not self.augmentation_majority_classes
        ):
            raise ValueError(
                "minority augmentation requires augmentation_majority_classes"
            )
        if (
            self.minority_augmentation_profile is None
            and self.augmentation_majority_classes
        ):
            raise ValueError(
                "augmentation_majority_classes requires minority augmentation"
            )
        self.minority_augment = (
            None
            if self.minority_augmentation_profile is None
            else clip_train_transform(
                image_size,
                self.minority_augmentation_profile,
            )
        )

    def augmentation_profile_for_label(self, label: int) -> str:
        if (
            self.minority_augmentation_profile is None
            or int(label) in self.augmentation_majority_classes
        ):
            return self.augmentation_profile
        return self.minority_augmentation_profile

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        image = Image.open(self.root / rec["relative_path"]).convert("RGB")
        sid = int(rec["sample_id"])
        label = int(rec["label"])
        augment = (
            self.augment
            if self.augmentation_profile_for_label(label)
            == self.augmentation_profile
            else self.minority_augment
        )
        return {
            "image": self.transform(image),
            "augmented_image": augment(image),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": torch.tensor(sid, dtype=torch.long),
            "caption_id": torch.tensor(sid, dtype=torch.long),
            "caption_embedding": self.caption_by_id[sid].float(),
            "bank_index": torch.tensor(index, dtype=torch.long),
        }


class DeterministicTrainDataset(Dataset):
    def __init__(self, manifest_path: str | Path, caption_cache_path: Optional[str | Path] = None, image_size: int = 224):
        manifest = load_manifest(manifest_path)
        self.root = Path(manifest["root"])
        self.records = manifest["splits"]["train"]
        self.labels = [int(x["label"]) for x in self.records]
        self.transform = clip_eval_transform(image_size)
        self.caption_by_id = None
        if caption_cache_path is not None:
            cache = torch.load(caption_cache_path, map_location="cpu", weights_only=False)
            self.caption_by_id = {int(sid): cache["embeddings"][i] for i, sid in enumerate(cache["sample_ids"])}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        image = Image.open(self.root / rec["relative_path"]).convert("RGB")
        item = {
            "image": self.transform(image),
            "label": torch.tensor(int(rec["label"]), dtype=torch.long),
            "sample_id": torch.tensor(int(rec["sample_id"]), dtype=torch.long),
            "bank_index": torch.tensor(index, dtype=torch.long),
        }
        if self.caption_by_id is not None:
            item["caption_embedding"] = self.caption_by_id[int(rec["sample_id"])].float()
        return item


class ImageOnlyEvalDataset(Dataset):
    """Validation/test dataset whose public interface cannot carry text fields."""

    FORBIDDEN_FIELDS = {"caption", "caption_id", "caption_embedding", "text_input_ids", "text_encoder"}

    def __init__(self, manifest_path: str | Path, split: str, image_size: int = 224):
        if split not in {"validation", "test"}:
            raise ValueError("ImageOnlyEvalDataset is only for validation or test")
        manifest = load_manifest(manifest_path)
        self.root = Path(manifest["root"])
        self.records = manifest["splits"][split]
        self.labels = [int(x["label"]) for x in self.records]
        self.transform = clip_eval_transform(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        image = Image.open(self.root / rec["relative_path"]).convert("RGB")
        item = {
            "image": self.transform(image),
            "label": torch.tensor(int(rec["label"]), dtype=torch.long),
            "sample_id": torch.tensor(int(rec["sample_id"]), dtype=torch.long),
        }
        assert not (self.FORBIDDEN_FIELDS & set(item))
        return item


class TrainImageOnlyCalibrationDataset(Dataset):
    """Train-split images for fitting image-only calibration statistics.

    Unlike :class:`ImageOnlyEvalDataset`, this class has no split argument: its
    public interface is deliberately pinned to the frozen training split and
    cannot be redirected to validation or test data.
    """

    FORBIDDEN_FIELDS = ImageOnlyEvalDataset.FORBIDDEN_FIELDS

    def __init__(self, manifest_path: str | Path, image_size: int = 224):
        manifest = load_manifest(manifest_path)
        self.root = Path(manifest["root"])
        self.records = manifest["splits"]["train"]
        self.labels = [int(x["label"]) for x in self.records]
        self.transform = clip_eval_transform(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        rec = self.records[index]
        image = Image.open(self.root / rec["relative_path"]).convert("RGB")
        item = {
            "image": self.transform(image),
            "label": torch.tensor(int(rec["label"]), dtype=torch.long),
            "sample_id": torch.tensor(int(rec["sample_id"]), dtype=torch.long),
        }
        assert not (self.FORBIDDEN_FIELDS & set(item))
        return item


class PKBatchSampler(Sampler[List[int]]):
    def __init__(self, labels: Iterable[int], classes_per_batch: int, samples_per_class: int, seed: int = 0):
        self.labels = list(map(int, labels))
        self.p = int(classes_per_batch)
        self.k = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0
        self.by_class = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.by_class[label].append(index)
        if self.p > len(self.by_class):
            raise ValueError("classes_per_batch exceeds number of classes")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return max(1, len(self.labels) // (self.p * self.k))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        classes = sorted(self.by_class)
        for _ in range(len(self)):
            chosen = rng.sample(classes, self.p)
            batch = []
            for label in chosen:
                pool = self.by_class[label]
                batch.extend(rng.choices(pool, k=self.k) if len(pool) < self.k else rng.sample(pool, self.k))
            rng.shuffle(batch)
            yield batch

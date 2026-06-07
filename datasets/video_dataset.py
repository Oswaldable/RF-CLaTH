import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

from .transforms import labels_to_multihot, load_frames_from_directory, sample_or_pad_sequence


def _resolve_path(path: str, root: Optional[Path] = None) -> str:
    if not path:
        return path
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    return str(((root or Path.cwd()) / p).resolve())


def load_avhash_json(path: str, root: Optional[Path] = None) -> Dict:
    path = _resolve_path(path, root)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_label_field(raw: str, label_offset: int = 0) -> List[int]:
    """Parse a single-label or multi-label field into zero-based class ids."""
    raw = raw.strip()
    if not raw:
        return []
    for sep in [";", "|", " "]:
        raw = raw.replace(sep, ",")
    labels = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            labels.append(int(item) - label_offset)
    return labels


def read_split_list(
    list_file: str,
    dataset_name: str,
    num_classes: int,
    label_offset: int,
) -> List[Dict]:
    """Read AVHash ActivityNet/FCVID list files.

    ActivityNet lines are video_name,frame_count,class_label.
    FCVID lines are video_name,class_label.
    """
    records = []
    with open(list_file, "r", encoding="utf-8") as f:
        for row_index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            video_id = parts[0]
            if dataset_name in {"actnet", "activitynet"} and len(parts) >= 3:
                label_raw = ",".join(parts[2:])
            elif len(parts) >= 2:
                label_raw = ",".join(parts[1:])
            else:
                label_raw = ""
            labels = parse_label_field(label_raw, label_offset=label_offset)
            records.append(
                {
                    "video_id": video_id,
                    "labels": labels,
                    "target": labels_to_multihot(labels, num_classes),
                    "row_index": row_index,
                }
            )
    return records


class _LazyH5Reader:
    """Per-process H5 reader for DataLoader workers."""

    def __init__(self, feature_path: str):
        if h5py is None:
            raise ImportError("h5py is required to read pre-extracted H5 features.")
        self.feature_path = feature_path
        self.handle = None

    def _open(self):
        if self.handle is None:
            self.handle = h5py.File(self.feature_path, "r")
        return self.handle

    def get(self, key: str) -> torch.Tensor:
        h5 = self._open()
        if key not in h5:
            raise KeyError(f"Video id {key!r} not found in {self.feature_path}")
        obj = h5[key]
        if isinstance(obj, h5py.Group):
            if "vectors" in obj:
                arr = obj["vectors"][...]
            elif "features" in obj:
                arr = obj["features"][...]
            else:
                first_key = next(iter(obj.keys()))
                arr = obj[first_key][...]
        else:
            arr = obj[...]
        return torch.as_tensor(arr, dtype=torch.float32)

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class AVHashFeatureDataset(Dataset):
    """Dataset compatible with AVHash H5 features and split files.

    Returns:
        dict with video [25, D] or frames [25, 3, H, W], label [C],
        video_id string, and index int.
    """

    def __init__(
        self,
        data_cfg: Dict,
        split: str,
        input_type: str = "features",
        num_frames: int = 25,
        project_root: Optional[Path] = None,
        image_size: int = 224,
    ):
        self.data_cfg = dict(data_cfg)
        self.split = split
        self.input_type = input_type
        self.num_frames = num_frames
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.image_size = image_size

        if "avhash_config" in self.data_cfg and self.data_cfg["avhash_config"]:
            avhash_path = Path(_resolve_path(self.data_cfg["avhash_config"], self.project_root))
            if avhash_path.exists():
                avhash_cfg = load_avhash_json(str(avhash_path))
                self.data_cfg = {**avhash_cfg, **self.data_cfg}

        self.dataset_name = self.data_cfg.get("dataset", self.data_cfg.get("name", "fcvid"))
        self.num_classes = int(self.data_cfg.get("num_classes", self.data_cfg.get("num_class", 0)))
        self.label_offset = int(self.data_cfg.get("label_offset", 1 if self.dataset_name == "fcvid" else 0))
        self.cache_in_memory = bool(self.data_cfg.get("cache_in_memory", False))

        list_key = {
            "train": "train_list",
            "val": "val_list",
            "test": "test_list",
            "retrieval": "retrieval_list",
        }[split]
        self.list_file = _resolve_path(self.data_cfg[list_key], self.project_root)
        self.records = read_split_list(
            self.list_file,
            dataset_name=self.dataset_name,
            num_classes=self.num_classes,
            label_offset=self.label_offset,
        )

        self.feature_path = _resolve_path(
            self.data_cfg.get("feature_path") or self.data_cfg.get("Image", ""),
            self.project_root,
        )
        self.frame_root = _resolve_path(self.data_cfg.get("frame_root", ""), self.project_root)
        self._h5_reader = None
        self._feature_cache = {}
        if self.input_type == "features" and self.cache_in_memory:
            self._load_feature_cache()

    def _load_feature_cache(self):
        reader = _LazyH5Reader(self.feature_path)
        for record in self.records:
            self._feature_cache[record["video_id"]] = reader.get(record["video_id"])
        reader.close()

    def _reader(self) -> _LazyH5Reader:
        if self._h5_reader is None:
            self._h5_reader = _LazyH5Reader(self.feature_path)
        return self._h5_reader

    def _load_features(self, video_id: str) -> torch.Tensor:
        if self.cache_in_memory:
            x = self._feature_cache[video_id]
        else:
            x = self._reader().get(video_id)
        if x.ndim == 1:
            x = x.unsqueeze(0).expand(self.num_frames, -1)
        return sample_or_pad_sequence(x.float(), self.num_frames)

    def _load_frames(self, video_id: str) -> torch.Tensor:
        if not self.frame_root:
            raise ValueError("frame_root must be set when model.input_type='frames'.")
        video_dir = Path(video_id)
        if not video_dir.is_absolute():
            video_dir = Path(self.frame_root) / video_id
        return load_frames_from_directory(video_dir, self.num_frames, image_size=self.image_size)

    def __getitem__(self, index: int) -> Dict:
        record = self.records[index]
        video_id = record["video_id"]
        if self.input_type == "features":
            video = self._load_features(video_id)
        elif self.input_type == "frames":
            video = self._load_frames(video_id)
        else:
            raise ValueError(f"Unsupported input_type: {self.input_type}")
        return {
            "video": video,
            "label": record["target"].clone(),
            "video_id": video_id,
            "index": torch.tensor(index, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_reader"] = None
        return state

    def __del__(self):  # pragma: no cover
        if getattr(self, "_h5_reader", None) is not None:
            self._h5_reader.close()


class _LazyIndexedH5Reader:
    """Per-process H5 reader for array-style feature files."""

    def __init__(self, feature_path: str, feature_key: str = "feats"):
        if h5py is None:
            raise ImportError("h5py is required to read pre-extracted H5 features.")
        self.feature_path = feature_path
        self.feature_key = feature_key
        self.handle = None

    def _open(self):
        if self.handle is None:
            self.handle = h5py.File(self.feature_path, "r")
        return self.handle

    def __len__(self) -> int:
        return int(self._open()[self.feature_key].shape[0])

    def get(self, index: int) -> torch.Tensor:
        arr = self._open()[self.feature_key][int(index)]
        return torch.as_tensor(arr, dtype=torch.float32)

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class S5VHFeatureDataset(Dataset):
    """Dataset adapter for official S5VH array-style H5 features.

    S5VH ActivityNet stores features as one dense dataset:
        train_feats.h5: feats [N_train, 30, 2048]
        query_feats.h5: feats [N_query, 30, 2048]
        test_feats.h5: feats [N_retrieval, 30, 2048]

    Labels are needed only for query/retrieval evaluation. They can be provided
    as preconverted torch tensors because scipy may not be installed in the
    training environment.
    """

    def __init__(
        self,
        data_cfg: Dict,
        split: str,
        input_type: str = "features",
        num_frames: int = 30,
        project_root: Optional[Path] = None,
    ):
        if input_type != "features":
            raise ValueError("S5VHFeatureDataset supports input_type='features' only.")
        self.data_cfg = dict(data_cfg)
        self.split = split
        self.input_type = input_type
        self.num_frames = int(num_frames)
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.num_classes = int(self.data_cfg.get("num_classes", 200))
        self.feature_key = self.data_cfg.get("feature_key", "feats")

        feature_key = {
            "train": "train_feature_path",
            "val": "val_feature_path",
            "test": "test_feature_path",
            "retrieval": "retrieval_feature_path",
        }[split]
        label_key = {
            "train": "train_label_path",
            "val": "val_label_path",
            "test": "test_label_path",
            "retrieval": "retrieval_label_path",
        }[split]
        self.feature_path = _resolve_path(self.data_cfg[feature_key], self.project_root)
        self.label_path = _resolve_path(self.data_cfg.get(label_key, ""), self.project_root)
        self._h5_reader = None
        if h5py is None:
            raise ImportError("h5py is required to read pre-extracted H5 features.")
        with h5py.File(self.feature_path, "r") as h5:
            self.length = int(h5[self.feature_key].shape[0])
        self.labels = self._load_labels(self.label_path)

    def _load_labels(self, label_path: str) -> Optional[torch.Tensor]:
        if not label_path:
            return None
        suffix = Path(label_path).suffix.lower()
        if suffix in {".pt", ".pth"}:
            labels = torch.load(label_path, map_location="cpu")
            if isinstance(labels, dict):
                labels = labels["labels"] if "labels" in labels else labels["label"]
            return labels.float()
        raise ValueError(
            f"Unsupported S5VH label file {label_path!r}. "
            "Convert q_label.mat/re_label.mat to .pt first."
        )

    def _reader(self) -> _LazyIndexedH5Reader:
        if self._h5_reader is None:
            self._h5_reader = _LazyIndexedH5Reader(self.feature_path, feature_key=self.feature_key)
        return self._h5_reader

    def __len__(self) -> int:
        if self.labels is not None and self.split != "train":
            return int(self.labels.shape[0])
        return self.length

    def __getitem__(self, index: int) -> Dict:
        video = sample_or_pad_sequence(self._reader().get(index).float(), self.num_frames)
        if self.labels is None:
            label = torch.zeros(self.num_classes, dtype=torch.float32)
        else:
            label = self.labels[index].clone().float()
        return {
            "video": video,
            "label": label,
            "video_id": f"{self.split}_{index:06d}",
            "index": torch.tensor(index, dtype=torch.long),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_reader"] = None
        return state

    def __del__(self):  # pragma: no cover
        if getattr(self, "_h5_reader", None) is not None:
            self._h5_reader.close()


def select_dataset_config(cfg: Dict, dataset_name: Optional[str] = None) -> Dict:
    data_cfg = cfg["data"]
    dataset_name = dataset_name or data_cfg.get("name", "fcvid")
    selected = dict(data_cfg["datasets"][dataset_name])
    selected["cache_in_memory"] = data_cfg.get("cache_in_memory", False)
    return selected


def build_dataloader(
    cfg: Dict,
    split: str,
    dataset_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    project_root: Optional[Path] = None,
) -> DataLoader:
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    selected = select_dataset_config(cfg, dataset_name=dataset_name)
    dataset_format = selected.get("format", "avhash")
    dataset_cls = S5VHFeatureDataset if dataset_format == "s5vh_h5" else AVHashFeatureDataset
    dataset = dataset_cls(
        selected,
        split=split,
        input_type=model_cfg.get("input_type", "features"),
        num_frames=int(model_cfg.get("num_frames", 25)),
        project_root=project_root,
    )
    if batch_size is None:
        if split == "train":
            batch_size = int(train_cfg.get("batch_size", 64))
        else:
            batch_size = int(cfg.get("eval", {}).get("batch_size", train_cfg.get("batch_size", 64)))
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=split == "train",
    )


def build_dataloaders(
    cfg: Dict,
    dataset_name: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = build_dataloader(cfg, "train", dataset_name, project_root=project_root)
    val_loader = build_dataloader(cfg, "val", dataset_name, shuffle=False, project_root=project_root)
    retrieval_loader = build_dataloader(cfg, "retrieval", dataset_name, shuffle=False, project_root=project_root)
    return train_loader, val_loader, retrieval_loader

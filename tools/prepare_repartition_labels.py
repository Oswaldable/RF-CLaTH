import argparse
from pathlib import Path

import torch

try:
    import scipy.io as sio
except ImportError as exc:  # pragma: no cover
    sio = None
    _SCIPY_ERROR = exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]


LABEL_SPECS = [
    ("activitynet/q_label.mat", "q_label", "repartition_s5vh_activitynet_q_label.pt"),
    ("activitynet/re_label.mat", "re_label", "repartition_s5vh_activitynet_re_label.pt"),
    ("hmdb/hmdb_train_labels.mat", "labels", "repartition_s5vh_hmdb_train_label.pt"),
    ("hmdb/hmdb_train_labels.mat", "labels", "repartition_s5vh_hmdb_re_label.pt"),
    ("hmdb/hmdb_test_labels.mat", "labels", "repartition_s5vh_hmdb_q_label.pt"),
    ("ucf/ucf_train_labels.mat", "labels", "repartition_s5vh_ucf_train_label.pt"),
    ("ucf/ucf_train_labels.mat", "labels", "repartition_s5vh_ucf_re_label.pt"),
    ("ucf/ucf_test_labels.mat", "labels", "repartition_s5vh_ucf_q_label.pt"),
    ("fcv/fcv_train_labels.mat", "labels", "repartition_s5vh_fcv_train_label.pt"),
    ("fcv/fcv_test_labels.mat", "labels", "repartition_s5vh_fcv_re_label.pt"),
    ("fcv/fcv_query_labels.mat", "labels", "repartition_s5vh_fcv_q_label.pt"),
]


def _resolve(path: str, root: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def main():
    parser = argparse.ArgumentParser(description="Convert dataset_rePartition MATLAB labels to RF-CLaTH .pt labels.")
    parser.add_argument("--source-root", default="/mnt/disk2/yql/dataset_rePartition")
    parser.add_argument("--output-dir", default="cache")
    args = parser.parse_args()

    if sio is None:
        raise ImportError("scipy is required to read MATLAB .mat labels.") from _SCIPY_ERROR

    source_root = _resolve(args.source_root, PROJECT_ROOT)
    output_dir = _resolve(args.output_dir, PROJECT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    for relative_mat, key, output_name in LABEL_SPECS:
        mat_path = source_root / relative_mat
        data = sio.loadmat(mat_path)
        if key not in data:
            raise KeyError(f"{key!r} not found in {mat_path}")
        labels = torch.as_tensor(data[key]).float()
        output_path = output_dir / output_name
        torch.save({"labels": labels}, output_path)
        print(f"saved={output_path} shape={tuple(labels.shape)} source={mat_path}:{key}")


if __name__ == "__main__":
    main()

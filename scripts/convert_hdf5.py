#!/usr/bin/env python3
"""
Convert ann-benchmarks HDF5 datasets to MetalANNS .annbin format.

USAGE:
  python3 scripts/convert_hdf5.py --input sift-128-euclidean.hdf5 --output sift-128.annbin
  python3 scripts/convert_hdf5.py --input gist-960-euclidean.hdf5 --output gist-960.annbin --metric l2

HDF5 SCHEMA (ann-benchmarks.com format):
  /train      float32[N x D]   training vectors
  /test       float32[Q x D]   query vectors
  /neighbors  int32[Q x K]     ground-truth indices
  /distances  float32[Q x K]   ground-truth distances (unused)

DEPENDENCIES:
  pip install h5py numpy
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import h5py
import numpy as np

METRIC_RAW = {
    "cosine": 0,
    "l2": 1,
    "innerproduct": 2,
}


def infer_metric(filename: str) -> str:
    lower = filename.lower()
    if "euclidean" in lower:
        return "l2"
    if "angular" in lower:
        return "cosine"
    return "cosine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ann-benchmarks HDF5 to .annbin")
    parser.add_argument("--input", required=True, help="Input HDF5 path")
    parser.add_argument("--output", required=True, help="Output .annbin path")
    parser.add_argument(
        "--metric",
        choices=["cosine", "l2", "innerproduct"],
        default=None,
        help="Distance metric override (default: infer from filename)",
    )
    return parser.parse_args()


def require_dataset(group: h5py.File, key: str) -> np.ndarray:
    if key not in group:
        raise ValueError(f"Missing required dataset '/{key}'")
    return np.asarray(group[key])


def validate_schema(train: np.ndarray, test: np.ndarray, neighbors: np.ndarray, distances: np.ndarray) -> None:
    if train.ndim != 2:
        raise ValueError(f"/train must be rank-2; got shape {train.shape}")
    if test.ndim != 2:
        raise ValueError(f"/test must be rank-2; got shape {test.shape}")
    if neighbors.ndim != 2:
        raise ValueError(f"/neighbors must be rank-2; got shape {neighbors.shape}")
    if distances.ndim != 2:
        raise ValueError(f"/distances must be rank-2; got shape {distances.shape}")

    train_count, dim = train.shape
    test_count, test_dim = test.shape
    n_q, k = neighbors.shape
    d_q, d_k = distances.shape

    if train_count <= 0 or test_count <= 0 or dim <= 0:
        raise ValueError("/train and /test must be non-empty with positive dimension")
    if test_dim != dim:
        raise ValueError(f"Dimension mismatch: train dim {dim}, test dim {test_dim}")
    if n_q != test_count:
        raise ValueError(f"neighbors rows ({n_q}) must match test count ({test_count})")
    if d_q != test_count or d_k != k:
        raise ValueError(f"distances shape {distances.shape} must match neighbors shape {neighbors.shape}")


def write_annbin(
    output_path: Path,
    train: np.ndarray,
    test: np.ndarray,
    neighbors: np.ndarray,
    metric: str,
) -> None:
    train_f = np.asarray(train, dtype="<f4", order="C")
    test_f = np.asarray(test, dtype="<f4", order="C")
    gt_u = np.asarray(neighbors, dtype="<u4", order="C")

    train_count, dim = train_f.shape
    test_count, _ = test_f.shape
    neighbors_count = gt_u.shape[1]

    header = struct.pack(
        "<4sIIIIIIIII",
        b"ANNB",     # magic
        1,           # version
        train_count,
        test_count,
        dim,
        neighbors_count,
        METRIC_RAW[metric],
        0,
        0,
        0,
    )

    with output_path.open("wb") as f:
        f.write(header)
        f.write(train_f.tobytes(order="C"))
        f.write(test_f.tobytes(order="C"))
        f.write(gt_u.tobytes(order="C"))


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    metric = args.metric or infer_metric(input_path.name)

    try:
        with h5py.File(input_path, "r") as h5:
            train = require_dataset(h5, "train")
            test = require_dataset(h5, "test")
            neighbors = require_dataset(h5, "neighbors")
            distances = require_dataset(h5, "distances")

            validate_schema(train, test, neighbors, distances)
            write_annbin(output_path, train, test, neighbors, metric)

        train_count, dim = train.shape
        test_count, _ = test.shape
        neighbors_count = neighbors.shape[1]
        print(
            f"Written {train_count}x{dim} train + {test_count}x{dim} test, "
            f"k={neighbors_count}, {metric} -> {output_path}"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

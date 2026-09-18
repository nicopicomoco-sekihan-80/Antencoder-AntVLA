import glob
import os

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

import tensorflow as tf


class BridgeActionDataset(IterableDataset):
    """
    BridgeData V2 TFRecordからaction windowをストリーミングするDataset。

    全trajectoryをRAMに保持せず、
    TFRecord -> trajectory -> action window
    の順に逐次読み込む。

    Returns:
        action: [seq_len, 7]
    """

    def __init__(
        self,
        root_dir,
        seq_len=32,
        stride=16,
        max_trajectories=None,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.seq_len = seq_len
        self.stride = stride
        self.max_trajectories = max_trajectories

        pattern = os.path.join(
            root_dir,
            "**",
            "bridge_dataset-*.tfrecord-*",
        )

        self.files = sorted(
            glob.glob(pattern, recursive=True)
        )

        if not self.files:
            raise FileNotFoundError(
                f"No BridgeData V2 TFRecord files found under: {root_dir}"
            )

        print(f"Found TFRecord shards: {len(self.files)}")

    def _parse_record(self, raw_record):
        example = tf.train.Example.FromString(
            raw_record.numpy()
        )

        features = example.features.feature

        action = np.asarray(
            features["steps/action"].float_list.value,
            dtype=np.float32,
        )

        reward = np.asarray(
            features["steps/reward"].float_list.value,
            dtype=np.float32,
        )

        if len(reward) == 0:
            raise ValueError("No reward/step count found")

        T = len(reward)

        if action.size % T != 0:
            raise ValueError(
                f"Action size {action.size} is not divisible by T={T}"
            )

        action = action.reshape(T, -1)

        if action.shape[1] != 7:
            raise ValueError(
                f"Expected action dim 7, got {action.shape}"
            )

        return action

    def _iter_files_for_worker(self):
        """
        DataLoaderのworkerごとにshardを分割する。
        workerがない場合は全shardを使用。
        """

        worker_info = get_worker_info()

        if worker_info is None:
            return self.files

        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        return self.files[worker_id::num_workers]

    def __iter__(self):
        files = self._iter_files_for_worker()

        trajectory_count = 0
        window_count = 0

        for shard_path in files:

            dataset = tf.data.TFRecordDataset(
                shard_path
            )

            for raw_record in dataset:

                if (
                    self.max_trajectories is not None
                    and trajectory_count >= self.max_trajectories
                ):
                    print(
                        f"Found trajectories: {trajectory_count}"
                    )
                    print(
                        f"Action windows: {window_count}"
                    )
                    return

                try:
                    actions = self._parse_record(
                        raw_record
                    )

                    trajectory_count += 1

                    T = len(actions)

                    if T < self.seq_len:
                        continue

                    for start in range(
                        0,
                        T - self.seq_len + 1,
                        self.stride,
                    ):
                        window = actions[
                            start : start + self.seq_len
                        ]

                        window_count += 1

                        yield torch.from_numpy(
                            window
                        )

                except Exception as e:
                    print(
                        f"[SKIP] {shard_path}"
                    )
                    print(
                        f"       {e}"
                    )

        print(
            f"Found trajectories: {trajectory_count}"
        )
        print(
            f"Action windows: {window_count}"
        )

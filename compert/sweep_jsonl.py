# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import json
import sys

import submitit
from compert.train import parse_arguments, train_compert

if __name__ == "__main__":
    json_file = sys.argv[1]

    with open(json_file, "r") as f:
        commands = [json.loads(line) for line in f.readlines()]

    executor = submitit.SlurmExecutor(folder="/checkpoint/dlp/sweep_jsonl/")

    executor.update_parameters(
        time=3 * 24 * 60,
        gpus_per_node=1,
        array_parallelism=512,
        cpus_per_task=4,
        comment="Deadline nat biotech this week",
        partition="priority",
    )

    executor.map_array(train_compert, commands)

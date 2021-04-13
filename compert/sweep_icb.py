# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import submitit
from compert.train import train_compert, parse_arguments


if __name__ == "__main__":
    args = parse_arguments()

    executor = submitit.SlurmExecutor(folder=args["save_dir"])

    executor.update_parameters(
        time=2 * 24 * 60,
        gpus_per_node=1,
        array_parallelism=60,
        cpus_per_task=4,
        partition="gpu_p",
        additional_parameters={'qos':'gpu'}
    )

    commands = []
    for seed in range(args["sweep_seeds"]):
        these_args = dict(args)
        these_args["seed"] = seed
        commands.append(these_args)

    executor.map_array(train_compert, commands)

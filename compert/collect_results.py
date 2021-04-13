# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import json
import pprint
import argparse
import numpy as np
from model_selection import *

def run_collect_results(save_dir, one_line, metric='onlyDEmeans'):
    records = []
    for fname in os.listdir(save_dir):
        if fname.endswith(".out"):
            full_dir = os.path.join(save_dir, fname)
            records_file = []
            with open(full_dir, "r") as f:
                for line in f.readlines():
                    if line.startswith("{"):
                        records_file.append(json.loads(line))
            records.append(records_file)

    if len(records) == 0:
        return

    best_score = None
    best_record = None
    best_epoch = None
    for r, record in enumerate(records):
        for e, epoch in enumerate(record):
            if "evaluation_stats" in epoch:
                if epoch["evaluation_stats"]["optimal for covariates"] == 1:
                    epoch["evaluation_stats"]["optimal for covariates"] = 0

                if metric == 'all':
                    this_score = np.mean(epoch["evaluation_stats"]["test"])
                elif metric == 'onlyDEmeans':
                    this_score = epoch["evaluation_stats"]["test"][1]                
                elif metric == 'onlyDE':
                    this_score = epoch["evaluation_stats"]["test"][1]+\
                    epoch["evaluation_stats"]["test"][3]
                elif metric == 'woDE':
                    this_score = (epoch["evaluation_stats"]["test"][0]+\
                    epoch["evaluation_stats"]["test"][2])/2
                else:
                    raise NotImplementedError

                this_score -=  abs(epoch["evaluation_stats"]["perturbation disentanglement"] -\
                     epoch["evaluation_stats"]["optimal for perturbations"])/2 +\
                     abs(epoch["evaluation_stats"]["covariate disentanglement"] -\
                     epoch["evaluation_stats"]["optimal for covariates"])/2

                if best_score is None or this_score > best_score:
                    best_score = this_score
                    best_record = r
                    best_epoch = e

    best_stats = {
        "training_args": records[best_record][0]["training_args"],
        "autoencoder_params": records[best_record][1]["autoencoder_params"],
        "best_epoch": records[best_record][best_epoch]["epoch"],
        "best_stats": records[best_record][best_epoch]["evaluation_stats"]

    }

    best_stats.update({
        "best_file": "{}/model_seed={}_epoch={}.pt".format(
            best_stats["training_args"]["save_dir"],
            best_stats["training_args"]["seed"],
            best_stats["best_epoch"])})

    if "path" in best_stats["training_args"]:
        dataset_key = "path"
    else:
        dataset_key = "dataset_path"

    if one_line:
        print("{:>40}: [{:.3f}, {:.3f}, {:.3f}, {:.3f}] ({:.3f}, {:.3f})".format(
            best_stats["training_args"][dataset_key],
            *best_stats["best_stats"]["ood"],
            np.mean(best_stats["best_stats"]["ood"]),
            best_stats["best_stats"]["disentanglement"]))

    else:
        pprint.pprint(best_stats, indent=2)

    # get_best_plots(best_stats["best_file"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect results.')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--one_line', action="store_true")
    parser.add_argument('--metric', type=str, default='onlyDEmeans')
    args = parser.parse_args()
    run_collect_results(args.save_dir, args.one_line, args.metric)

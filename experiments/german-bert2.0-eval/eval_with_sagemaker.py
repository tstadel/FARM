# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Downstream runner for all experiments in specified config files."""

from pathlib import Path
from experiment import run_experiment, load_experiments
from convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch as convert_tf_to_pt_bert
from convert_albert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch as convert_tf_to_pt_albert
from convert_electra_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch as convert_tf_to_pt_electra

import os
from shutil import copyfile
import torch
import json
import logging

CONFIG_FILES = {
    "germEval18Fine": Path("experiments/german-bert2.0-eval/germEval18Fine_config.json"),
    "germEval18Coarse": Path("experiments/german-bert2.0-eval/germEval18Coarse_config.json"),
    "germEval14": Path("experiments/german-bert2.0-eval/germEval14_config.json")
}
COMPLETED_FILE = "/opt/ml/checkpoints/completed.txt"

def fetch_tf_checkpoints(dir):
    files = os.listdir(dir)
    files = [f for f in files if "model.ckpt-" in f]
    checkpoints = set(".".join(f.split(".")[:2]) for f in files)
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]), reverse=True)
    return checkpoints

def fetch_pt_checkpoints(dir, model_type):
    files = os.listdir(dir)
    files = [dir / f for f in files if "pt_" in f]
    checkpoints = sorted(files, key=lambda x: int(str(x).replace(f"pt_{model_type}_", "").split("/")[-1].split("-")[0]), reverse=True)
    return checkpoints

def main_from_saved(args, completed):
    checkpoints = fetch_pt_checkpoints(Path(args["checkpoints_folder"]), args["model_type"])
    logging.info(f"Performing evaluation on these checkpoints: {checkpoints}")
    logging.info(f"Performing evaluation using these experiments: {CONFIG_FILES}")
    if completed:
        logging.info(f"Last completed: {completed[-1]}")
        logging.debug(f"completed: {completed}")
    for checkpoint in checkpoints:
        for i, (conf_name, conf_file) in enumerate(CONFIG_FILES.items()):
            experiments = load_experiments(conf_file)
            steps = str(checkpoint).split("_")[-1]
            for j, experiment in enumerate(experiments):
                mlflow_run_name = f"{conf_name}_step{steps}_{j}"
                if mlflow_run_name in completed:
                    continue
                experiment.logging.mlflow_url = args["mlflow_url"]
                experiment.logging.mlflow_experiment = args["mlflow_experiment"]
                experiment.logging.mlflow_run_name = mlflow_run_name
                experiment.parameter.model = checkpoint
                experiment.general.output_dir = str(checkpoint).split("/")[:-1]
                run_experiment(experiment)
                f = open(COMPLETED_FILE, "a")
                f.write(mlflow_run_name + "\n")
                f.close()
                torch.cuda.empty_cache()

# def main_from_downloaded(args, completed):
#     print(f"Performing evaluation on these models: {args.models}")
#     print(f"Performing evaluation using these experiments: {CONFIG_FILES}")
#     if completed:
#         logging.info(f"Last completed: {completed[-1]}")
#     for model in args.models:
#         for i, (conf_name, conf_file) in enumerate(CONFIG_FILES.items()):
#             experiments = load_experiments(conf_file)
#             for j, experiment in enumerate(experiments):
#                 mlflow_run_name = f"{conf_name}_{model}_{j}"
#                 if mlflow_run_name in completed:
#                     continue
#                 experiment.logging.mlflow_url = args.mlflow_url
#                 experiment.logging.mlflow_experiment = args.mlflow_experiment
#                 experiment.logging.mlflow_run_name = mlflow_run_name
#                 experiment.parameter.model = model
#                 experiment.general.output_dir = "benchmarks"
#                 run_experiment(experiment)
#                 open(COMPLETED_FILE, "a").write(mlflow_run_name + "\n").close()
#                 torch.cuda.empty_cache()

def main():
    with open("/opt/ml/input/config/hyperparameters.json") as f:
        params = json.load(f)
    logging.info(params)
    open(COMPLETED_FILE, "a").close()
    completed = [l.strip() for l in open(COMPLETED_FILE)]
    logging.info(f"Starting a train job with parameters {params}")
    if params["model_source"] == "checkpoints":
        main_from_saved(params, completed)
    # elif params["model_source"] == "download":
    #     main_from_downloaded(params, completed)
    #

if __name__ == "__main__":
    main()

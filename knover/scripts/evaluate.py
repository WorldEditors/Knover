#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""Traine main program."""

import argparse
from collections import defaultdict
import json
import os
import subprocess
import time

import paddle
import numpy as np
import paddle.fluid as fluid
from paddle.distributed import fleet

import knover.models as models
import knover.tasks as tasks
from knover.utils import check_cuda, parse_args, str2bool, Timer

writer=None
use_vdl=False

def setup_args():
    """Setup training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_vdl", type=str2bool, default=False,
                        help="Whether to run on vdl.")
    parser.add_argument("--on_k8s", type=str2bool, default=False,
                        help="Whether it is running on k8s.")
    parser.add_argument("--is_distributed", type=str2bool, default=False,
                        help="Whether to run distributed training.")
    parser.add_argument("--save_path", type=str, default="output",
                        help="The path where to save models.")
    parser.add_argument("--train_file", type=str, required=True,
                        help="The training dataset: file / filelist. "
                        "See more details in `docs/usage.md`: `file_format`.")
    parser.add_argument("--valid_file", type=str, required=True,
                        help="The validation datasets: files / filelists. "
                        "The files / filelists are separated by `,`. "
                        "See more details in `docs/usage.md`: `file_format`.")
    parser.add_argument("--vdl_log_path", type=str, default="vdl_logs",
                        help="The path where to put vdl_logs.")

    parser.add_argument("--start_step", type=int, default=0,
                        help="The start step of training. It will be updated if you load from a checkpoint.")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="The number of times that the learning algorithm will work through the entire training dataset.")
    parser.add_argument("--log_steps", type=int, default=100,
                        help="Display training / evaluation log information every X steps.")
    parser.add_argument("--validation_steps", type=int, default=1000,
                        help="Run validation every X training steps.")
    parser.add_argument("--save_steps", type=int, default=0,
                        help="Save the latest model every X training steps. "
                        "If `save_steps = 0`, then it only keep the latest checkpoint.")
    parser.add_argument("--eval_metric", type=str, default="-loss",
                        help="Keep the checkpoint with best evaluation metric.")
    parser.add_argument("--save_checkpoint", type=str2bool, default=True,
                        help="Save completed checkpoint or parameters only. "
                        "The checkpoint contains all states for continuous training.")

    models.add_cmdline_args(parser)
    tasks.add_cmdline_args(parser)

    args = parse_args(parser)
    args.load(args.config_path, "Model")
    args.display()
    return args


def run_cmd(cmd):
    """Helpful function for running shell command in py scripts."""
    exitcode, output = subprocess.getstatusoutput(cmd)
    return output


def train(args):
    """The main function of training."""
    if args.is_distributed:
        strategy = fleet.DistributedStrategy()
        strategy.find_unused_parameters = True
        fleet.init(strategy=strategy, is_collective=True)

        dev_count = fluid.core.get_cuda_device_count()
        gpu_id = int(os.getenv("FLAGS_selected_gpus"))
        trainers_num = fleet.worker_num()
        trainer_id = fleet.worker_index()
    else:
        dev_count = 1
        gpu_id = 0
        trainers_num = 1
        trainer_id = 0
    place = fluid.CUDAPlace(gpu_id)

    # setup task and model
    task = tasks.create_task(args)
    model = models.create_model(args, place)

    valid_tags = []
    valid_generators = []
    for valid_file in args.valid_file.split(","):
        if ":" in valid_file:
            valid_tag, valid_file = valid_file.split(":")
        else:
            valid_tag = "valid"
        valid_tags.append(valid_tag)
        valid_generators.append(task.get_data_loader(
            model,
            input_file=valid_file,
            num_part=dev_count,
            part_id=gpu_id,
            phase="distributed_valid" if args.is_distributed else "valid"
        ))

    for valid_tag, valid_generator in zip(valid_tags, valid_generators):
        eval_metrics = evaluate(task, model, valid_generator, args, dev_count, gpu_id, 0, tag=valid_tag)
        if valid_tag == "valid":
            valid_metrics = eval_metrics

    return


def evaluate(task,
             model,
             generator,
             args,
             dev_count,
             gpu_id,
             training_step,
             tag=None):
    """Run evaluation.

    Run evaluation on dataset which is generated from a generator. Support evaluation on single GPU and multiple GPUs.

    Single GPU:
    1. Run evaluation on the whole dataset (the generator generates the completed whole dataset).
    2. Disply evaluation result.

    Multiple GPUs:
    1. Each GPU run evaluation on a part of dataset (the generator only generate a part of dataset). The dataset
       is split into `dev_count` parts.
    2. Save evaluation results on each part of dataset.
    3. Merge all evaluation results into the final evaluation result.
    4. Save evaluation result on the whole dataset.
    5. Display evaluation result.
    """
    outputs = None
    print("=" * 80)
    print(f"Evaluation: {tag}")
    timer = Timer()
    timer.start()
    for step, data in enumerate(generator(), 1):
        part_outputs = task.eval_step(model, data)
        outputs = task.merge_metrics_and_statistics(outputs, part_outputs)

        if step % args.log_steps == 0:
            metrics = task.get_metrics(outputs)
            print(f"\tstep {step}:" + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    if args.is_distributed:
        # save part evaluation outputs in distributed mode.
        part_file = os.path.join(args.save_path, f"evaluation_output.part_{gpu_id}")
        with open(part_file, "w") as fp:
            json.dump(outputs, fp, ensure_ascii=False)
        part_finish_file = os.path.join(args.save_path, f"evaluation_output.part_{gpu_id}.finish")
        with open(part_finish_file, "w"):
            pass

        if gpu_id == 0:
            # wait part evaluation outputs
            part_files = f"evaluation_output.part_*.finish"
            while True:
                ret = run_cmd(f"find {args.save_path} -maxdepth 1 -name {part_files}")
                num_completed = len(ret.split("\n"))
                if num_completed == dev_count:
                    break
                time.sleep(1)

            # merge part evaluation outputs
            outputs = None
            for dev_id in range(dev_count):
                part_file = os.path.join(args.save_path, f"evaluation_output.part_{dev_id}")
                with open(part_file, "r") as fp:
                    part_outputs = json.load(fp)
                    outputs = task.merge_metrics_and_statistics(outputs, part_outputs)
            run_cmd("rm " + os.path.join(args.save_path, "evaluation_output.part_*"))

            # send evaluation outputs
            for dev_id in range(1, dev_count): # exclude gpu 0
                part_file = os.path.join(args.save_path, f"evaluation_output.final_part_{dev_id}")
                with open(part_file, "w") as fp:
                    json.dump(outputs, fp, ensure_ascii=False)
                with open(part_file + ".finish", "w") as fp:
                    pass
        else:
            # receive evaluation outputs
            part_file = os.path.join(args.save_path, f"evaluation_output.final_part_{gpu_id}")
            while not os.path.exists(part_file + ".finish"):
                time.sleep(1)
            with open(part_file, "r") as fp:
                outputs = json.load(fp)
            run_cmd(f"rm {part_file}*")

    metrics = task.get_metrics(outputs)
    print(f"[Evaluation][{training_step}] " + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))

    if use_vdl:
        for name, value in metrics.items():
            writer.add_scalar(f"{tag}/{name}", value, training_step)

    print(f"\ttime cost: {timer.pass_time:.3f}")
    print("=" * 80)
    return metrics


def save_model(model, save_path, tag, dev_count, gpu_id, args):
    """Save model.

    In normal mode, only the master GPU need to save the model.
    """
    path = os.path.join(save_path, tag)
    if gpu_id == 0:
        print(f"Saving model into {path}.")
        model.save(path, is_checkpoint=args.save_checkpoint)
        print(f"Model has saved into {path}.")

        # tar saved model
        run_cmd(f"cd {save_path} && tar --remove-files -cf {tag}.tar {tag}")
        print("Tar saved model (DONE)")
    return


if __name__ == "__main__":
    try:
        args = setup_args()
        check_cuda(True)
        train(args)
    except:
        if use_vdl and args.on_k8s:
            expt.fail()
        raise
    else:
        if use_vdl and args.on_k8s:
            expt.success()
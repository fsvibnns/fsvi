import argparse
import csv
import json
import logging
import os
import pickle
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import psutil
import torch
from retry import retry


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d-%A-%H-%M-%S")


def set_up_logging(log_path=None, loglevel=logging.DEBUG, stdout=False):
    logger = logging.getLogger(__name__)
    logger.setLevel(loglevel)
    logger.handlers = []

    hs = []

    if stdout:
        hs.append(logging.StreamHandler())
    if log_path is not None:
        hs.append(logging.FileHandler(log_path, mode="w"))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    for h in hs:
        h.setLevel(loglevel)
        h.setFormatter(formatter)
        logger.addHandler(h)
    # this line is to prevent logger to print to stdout
    logger.propagate = False
    return logger


PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
EXPS_ROOT = PROJECT_ROOT / "fsvi_cl/src_cl/exps"


def path_is_relative(path: str) -> bool:
    return "fsvi" not in str(path)


def make_path_absolute(path: str, root=EXPS_ROOT) -> str:
    if path_is_relative(path):
        return root / path
    else:
        return Path(path)


@retry(FileExistsError, tries=100, delay=1, backoff=1.5, jitter=(0.1, 0.5))
def create_logdir(rootdir=None, subdir=None):

    if (rootdir is None) or (rootdir == ""):
        p = Path.cwd()
    else:
        # TODO: check that the path exists? it will be created but perhaps print warning
        p = make_path_absolute(rootdir)

    timestamp = get_timestamp()

    if subdir is not None:
        p = p / "runs" / subdir / timestamp
    else:
        p = p / "runs" / timestamp

    p.mkdir(parents=True)

    save_repo_status(p)
    save_command_line(p)
    save_slurm_status(p)

    return p


# The retry logic is helpful for avoiding error on Slurm where multiple jobs could start simultaneously


def save_repo_status(path):
    with open(path / "git_commit.txt", "w") as f:
        subprocess.run(["git", "rev-parse", "HEAD"], stdout=f)

    with open(path / "workspace_changes.diff", "w") as f:
        subprocess.run(["git", "diff"], stdout=f)


def save_command_line(path):
    with open(path / "command_line.txt", "w") as f:
        f.write("python " + " ".join(sys.argv))


def oatml_slurm():
    return os.environ.get("SLURM_CLUSTER_NAME") == "oatcloud"


def oxmlsys_slurm():
    return os.environ.get("SLURM_CLUSTER_NAME") == "oxmlsys"


def arc_slurm():
    return os.environ.get("SLURM_CLUSTER_NAME") == "arcus-htc"


def is_azure():
    return os.environ.get("IS_AZURE") == "1"


def slurm_active():
    return oatml_slurm() or oxmlsys_slurm() or arc_slurm()


def oatml_workstation():
    return socket.gethostname().startswith("clpc")


def if_not_slurm(func):
    def wrapper(x):
        if not slurm_active():
            return func(x)
        else:
            return x

    return wrapper


def num_available_cpus():
    if slurm_active():
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        return psutil.cpu_count()


def num_available_gpus():
    return torch.cuda.device_count()


def running_in_slurm():
    job_id = os.environ.get("SLURM_JOB_ID")
    return job_id is not None


def save_slurm_status(path):
    job_id = os.environ.get("SLURM_JOB_ID")
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if array_job_id is not None:
        Path(path / f"slurm_job_{array_job_id}_{array_task_id}").touch()
    elif job_id is not None:
        Path(path / f"slurm_job_{job_id}").touch()


def save_kwargs(kwargs: Dict, path):
    config_header = kwargs.keys()
    with open(path, "a") as config_file:
        config_writer = csv.DictWriter(config_file, fieldnames=config_header)
        config_writer.writeheader()
        config_writer.writerow(kwargs)
        config_file.close()


class Hyperparameters:
    def __init__(self, **kwargs):
        self.from_dict(kwargs)

    def from_argparse(self, args: argparse.Namespace):
        self.from_dict(args.__dict__)

    def from_dict(self, d: Dict):
        for k, v in d.items():
            setattr(self, k, v)

    def from_json(self, j):
        d = json.loads(j)
        self.from_dict(d)

    def as_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dict__}

    def to_json(self, path):
        j = json.dumps(self.as_dict(), indent=4, sort_keys=True)
        path.write_text(j)

    def __contains__(self, k):
        return k in self.__dict__

    def __str__(self):
        # s = [f"{k}={v}" for k, v in self.as_dict().items()]
        # return ",".join(s)
        s = json.dumps(self.as_dict(), indent=4, separators=(",", ":"))
        return f"input args:\n {s}"


def save_chkpt(p, **kwargs):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, f"wb") as f:
        pickle.dump(kwargs, f)

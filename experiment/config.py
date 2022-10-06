import os
import pwd
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Optional, Dict, Tuple, Any, Type
from torch import nn
import torch
from torchvision.transforms import transforms

from util.tool_function import safe_dir


def transform_dict(config_dict: Dict, expand: bool = True):
    """
    General function to transform any dictionary into wandb config acceptable format
    (This is mostly due to datatypes that are not able to fit into YAML format which makes wandb angry)
    The expand argument is used to expand iterables into dictionaries
    So the configs can be used when comparing results across runs
    """
    ret: Dict[str, Any] = {}
    for k, v in config_dict.items():
        if v is None or isinstance(v, (int, float, str)):
            ret[k] = v
        elif isinstance(v, (list, tuple, set)):
            # Need to check if item in iterable is YAML-friendly
            t = transform_dict(dict(enumerate(v)), expand)
            # Transform back to iterable if expand is False
            ret[k] = t if expand else [t[i] for i in range(len(v))]
        elif isinstance(v, dict):
            ret[k] = transform_dict(v, expand)
        else:
            # Transform to YAML-friendly (str) format
            # Need to handle both Classes, Callables, Object Instances
            # Custom Classes might not have great __repr__ so __name__ might be better in these cases
            vname = v.__name__ if hasattr(v, '__name__') else v.__class__.__name__
            ret[k] = f"{v.__module__}:{vname}"
    return ret


def dfac_dataset_optimizer_args():
    return {
        "lr": 0.001,
        # "momentum": 0.9,
        # "weight_decay": 5e-4,
    }


def dfac_lr_scheduler_args():
    return {
        'step_size': 3,
        'gamma': 0.1,
    }


def dfac_cur_time():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


@dataclass
class ExperimentConfig:
    # Project Name
    project_name: str = "project_video_streaming_HW1"

    # Data Sources
    db_path: str = "/mnt/2ndHDD/oscarchencs10/video_streaming/"

    # GPU Device Setting
    gpu_device_id: str = "1"

    # tensorboard setting
    tensorboard_log_root: str = safe_dir(f"/home/oscarchencs10/video_streaming/HW1/log/tensorboard")

    # WandB setting
    wandb_repo: str = "project_VS_repo"
    wandb_project: str = "project_video_streaming_HW1"
    # wandb_group: str = "test"
    wandb_entity: str = "oscarchencs10"
    username = pwd.getpwuid(os.getuid()).pw_name
    wandb_dir: str = safe_dir(f"/tmp/video_streaming_wandb_{username}/")

    # Set random seed. Set to None to create new Seed
    random_seed: int = 42

    # TODO: (canceliation) Cross Validation Split
    # cv_split: int = 3

    # Training Related
    num_epochs: int = 20
    batch_size: int = 10
    dataloader_num_worker: int = 4
    # learning_rate: float = 0.001
    # momentum_val: float = 0.9
    # weight_decay_val: float = 5e-4
    patience: int = 2 # Early Stopping

    # Default Cross Entropy loss
    loss_function: nn.Module = nn.CrossEntropyLoss()

    # Default Don't Select Model
    # model: Optional[Type[nn.Module]] = None
    # model_args: Dict[str, Any] = field(default_factory=dict)

    # current time
    # cur_time: str = field(default_factory=dfac_cur_time)
    cur_time: str = dfac_cur_time()

    # Default model save root
    checkpoint_root: str = safe_dir(
        f"/home/oscarchencs10/{project_name}/{project_name + '_Code'}/model_saved/{cur_time}")

    # Default Select Adam as Optimizer
    optimizer: Type[torch.optim.Optimizer] = torch.optim.SGD  # type: ignore
    optimizer_args: Dict[str, Any] = field(default_factory=dfac_dataset_optimizer_args)

    # Default adjust learning rate
    lr_scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None  # pylint: disable=protected-access
    lr_scheduler_args: Dict[str, Any] = field(default_factory=dfac_lr_scheduler_args)

    # Transform
    train_data_transform = [transforms.Resize((224, 224)), transforms.ToTensor()]
    test_data_transform = [transforms.Resize((224, 224)), transforms.ToTensor()]

    # Log data store
    log_data_dir: str = f'/home/oscarchencs10/{project_name}/{project_name + "_Code"}/log/logging/{cur_time}_VS.log'

    def to_dict(self, expand: bool = True):
        return transform_dict(asdict(self), expand)


if __name__ == "__main__":
    config = ExperimentConfig()
    print(config)
    print(config.to_dict())

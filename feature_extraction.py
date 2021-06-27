__author__ = "shekkizh"
"""Modified code for feature extraction using VISSL tutorial and tools"""
import argparse
import os
from typing import Any, List
from vissl.utils.hydra_config import convert_to_attrdict, is_hydra_available
from hydra.experimental import compose, initialize_config_module
from vissl.utils.distributed_launcher import launch_distributed
from vissl.hooks import default_hook_generator
# from vissl.utils.env import set_env_vars
from vissl.data.dataset_catalog import VisslDatasetCatalog

parser = argparse.ArgumentParser(description='VISSL extract features')
parser.add_argument('--model_url',
                    default='https://dl.fbaipublicfiles.com/vissl/model_zoo/deepclusterv2_800ep_pretrain.pth.tar',
                    help='Model to download - https://github.com/facebookresearch/vissl/blob/master/MODEL_ZOO.md')
parser.add_argument('--logs_dir', default='/media/charlie/hd_1/VISSL')
parser.add_argument("--config", default="imagenet1k_resnet50_trunk_features.yaml",
                    help="config file to extract features")


def hydra_main(overrides: List[Any]):
    print(f"####### overrides: {overrides}")
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=overrides)
    args, config = convert_to_attrdict(cfg)
    # set_env_vars(local_rank=0, node_id=0, cfg=config)
    launch_distributed(
        cfg=config,
        node_id=args.node_id,
        engine_name=args.engine_name,
        hook_generator=default_hook_generator,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    print("Retrieving model weights from VISSL MODEL ZOO")
    basename = os.path.basename(args.model_url)
    weights_file = os.path.join('/media/charlie/HD_2/PyTorch_hub', basename)
    if not os.path.exists(weights_file):
        os.system(f"wget -O {weights_file}  -L {args.model_url}")

    logs_dir = os.path.join(args.logs_dir, basename.split('.')[0])

    # print imagenet path
    print(VisslDatasetCatalog.get("imagenet1k_folder"))

    overrides = [f"config={args.config}", f"config.CHECKPOINT.DIR={logs_dir}",
                 f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={weights_file}"]

    assert is_hydra_available(), "Make sure to install hydra"
    overrides.append("hydra.verbose=true")
    hydra_main(overrides=overrides)

__author__ = "shekkizh"
"""Modified code for feature extraction using VISSL tutorial and tools codes"""
import argparse
import os
import numpy as np
import torch, faiss
from typing import Any, List
from vissl.config import AttrDict
from vissl.utils.hydra_config import convert_to_attrdict, is_hydra_available
from hydra.experimental import compose, initialize_config_module
from vissl.utils.distributed_launcher import launch_distributed
from vissl.hooks import default_hook_generator
from vissl.models.model_helpers import get_trunk_output_feature_names
from vissl.utils.misc import merge_features
from vissl.utils.checkpoint import get_checkpoint_folder
from vissl.data.dataset_catalog import VisslDatasetCatalog

parser = argparse.ArgumentParser(description='VISSL extract features')
parser.add_argument('--model_url',
                    default='https://dl.fbaipublicfiles.com/vissl/model_zoo/deepclusterv2_800ep_pretrain.pth.tar',
                    help='Model to download - https://github.com/facebookresearch/vissl/blob/master/MODEL_ZOO.md')
parser.add_argument('--logs_dir', default='/media/charlie/hd_1/VISSL')
parser.add_argument("--config", default="imagenet1k_resnet50_trunk_features.yaml",
                    help="config file to extract features")
parser.add_argument('--top_k', default=50, help="initial no. of neighbors")


def nnk_classifier(features, labels, queries, targets, num_neighbors, num_classes=1000):
    return 0, 0


def benchmark_layer(cfg: AttrDict, layer_name: str = "heads"):
    num_neighbors = cfg.NEAREST_NEIGHBOR.TOPK
    output_dir = get_checkpoint_folder(cfg)

    train_out = merge_features(output_dir, "train", layer_name, cfg)
    train_features, train_labels = train_out["features"], train_out["targets"]

    num_classes = np.max(train_labels) + 1

    test_out = merge_features(output_dir, "test", layer_name, cfg)
    test_features, test_labels = test_out["features"], test_out["targets"]

    top1, top5 = nnk_classifier(train_features, train_labels, test_features, test_labels, num_neighbors, num_classes)
    return top1, top5


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

    feat_names = get_trunk_output_feature_names(config.MODEL)
    if len(feat_names) == 0:
        feat_names = ["heads"]

    for layer in feat_names:
        top1, top5 = benchmark_layer(config, layer_name=layer)


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
                 f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={weights_file}", f"config.NEAREST_NEIGHBOR.TOPK={args.top_k}"]

    assert is_hydra_available(), "Make sure to install hydra"
    overrides.append("hydra.verbose=true")
    hydra_main(overrides=overrides)

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
from utils.non_neg_qpsolver import non_negative_qpsolver

parser = argparse.ArgumentParser(description='VISSL extract features')
parser.add_argument('--model_url',
                    default='https://dl.fbaipublicfiles.com/vissl/model_zoo/deepclusterv2_800ep_pretrain.pth.tar',
                    help='Model to download - https://github.com/facebookresearch/vissl/blob/master/MODEL_ZOO.md')
parser.add_argument('--logs_dir', default='/scratch/shekkizh/logs/VISSL')
parser.add_argument("--config", default="imagenet1k_resnet50_trunk_features.yaml",
                    help="config file to extract features")
parser.add_argument('--top_k', default=50, help="initial no. of neighbors")
parser.add_argument('--extract_features', dest='extract_features', action='store_true')
parser.add_argument('--noextract_features', dest='extract_features', action='store_false')
parser.set_defaults(extract_features=False)

def to_categorical(y, num_classes=None, dtype='float32'):
    """
    Code taken from keras to categorical
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


@torch.no_grad()
def nnk_classifier(features, labels, queries, targets, topk, num_classes=1000):
    dim = features.shape[1]
    target_one_hot = to_categorical(labels, num_classes)
    normalized_features = features / np.linalg.norm(features, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(normalized_features)

    normalized_queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    n_queries = queries.shape[0]
    soft_prediction = np.zeros(shape=(n_queries, num_classes), dtype=np.float)

    distances, indices = index.search(normalized_queries, topk)

    for ii, x_test in enumerate(normalized_queries):
        neighbor_indices = indices[ii, :]
        neighbor_labels = target_one_hot[neighbor_indices, :]
        x_support = normalized_features[neighbor_indices]
        g_i = 0.5 + np.dot(x_support, x_test) / 2
        G_i = 0.5 + np.dot(x_support, x_support.T) / 2
        x_opt = non_negative_qpsolver(G_i, g_i, g_i, x_tol=1e-10)
        # x_opt = g_i
        non_zero_indices = np.nonzero(x_opt)
        x_opt = x_opt[non_zero_indices] / np.sum(x_opt[non_zero_indices])
        soft_prediction[ii, :] = np.dot(x_opt, neighbor_labels[non_zero_indices])
        if ii % 10000 == 0:
            print(f"{ii}/{n_queries} processed...")

    probs = torch.from_numpy(soft_prediction).cuda()
    targets = torch.from_numpy(targets).cuda()
    _, predictions = probs.sort(1, True)
    correct = predictions.eq(targets.data.view(-1, 1))
    top1 = correct.narrow(1, 0, 1).sum().item() * 100.0 / n_queries
    top5 = correct.narrow(1, 0, 5).sum().item() * 100.0 / n_queries
    return top1, top5


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


def hydra_main(overrides: List[Any], extract_features=False):
    print(f"####### overrides: {overrides}")
    with initialize_config_module(config_module="vissl.config"):
        cfg = compose("defaults", overrides=overrides)
    args, config = convert_to_attrdict(cfg)

    if extract_features:
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
        print(f"NNK classifier - Layer: {layer}, Top1: {top1}, Top5: {top5}")


if __name__ == "__main__":
    args = parser.parse_args()
    print("Retrieving model weights from VISSL MODEL ZOO")
    basename = os.path.basename(args.model_url)
    weights_file = os.path.join('/scratch/shekkizh/torch_hub/checkpoints/', basename)
    if not os.path.exists(weights_file):
        os.system(f"wget -O {weights_file}  -L {args.model_url}")

    logs_dir = os.path.join(args.logs_dir, basename.split('.')[0])

    # print imagenet path
    print(VisslDatasetCatalog.get("imagenet1k_folder"))

    overrides = [f"config={args.config}", f"config.CHECKPOINT.DIR={logs_dir}",
                 f"config.MODEL.WEIGHTS_INIT.PARAMS_FILE={weights_file}", f"config.NEAREST_NEIGHBOR.TOPK={args.top_k}"]

    assert is_hydra_available(), "Make sure to install hydra"
    overrides.append("hydra.verbose=true")
    hydra_main(overrides=overrides, extract_features=args.extract_features)

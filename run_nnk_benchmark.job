#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:25:00
#SBATCH --mem=0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:2

module load gcc
module load python/3.7.6
module load cuda/10.1.243
module load cudnn/8.0.2-10.1

source ~/vissl/bin/activate
# python nnk_benchmark.py --model_url /scratch/shekkizh/torch_hub/checkpoints/resnet50-19c8e357.pth --top_k 50 --noextract_features

# python nnk_benchmark.py --config imagenet1k_resnet50_deepclusterv2  --model_url /scratch/shekkizh/torch_hub/checkpoints/deepclusterv2_800ep_pretrain.pth.tar --top_k 50 --noextract_features

# python nnk_benchmark.py --config imagenet1k_resnet50_swav  --model_url /scratch/shekkizh/torch_hub/checkpoints/swav_rn50_model_final_checkpoint_phase799.torch --top_k 50 --noextract_features

## python nnk_benchmark.py --config imagenet1k_resnet50_swav  --model_url /scratch/shekkizh/torch_hub/checkpoints/mocov2_model_final_checkpoint_phase199.torch --top_k 50 --noextract_features
## using same config as swav for MoCov2

python nnk_benchmark.py --config imagenet1k_resnet50_mocov2_800ep.yaml  --model_url /scratch/shekkizh/torch_hub/checkpoints/moco_v2_800ep_pretrain.pth.tar --top_k 50 --noextract_features

# python nnk_benchmark.py --model_url /scratch/shekkizh/torch_hub/checkpoints/dino_resnet50_pretrain.pth --top_k 50 --noextract_features
# using same config as rn50 for DINO rn50 evaluation


#######################################
# Runs for other values of k

# python nnk_benchmark.py --model_url /scratch/shekkizh/torch_hub/checkpoints/resnet50-19c8e357.pth --top_k 20 --noextract_features

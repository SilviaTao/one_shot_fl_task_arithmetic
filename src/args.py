import os
import argparse

import torch
from src.utils import WORK_DIR

# #WORK_DIR = '/home/group/self_improving/experiments/mixing'
# WORK_DIR = '/groups/gcd50678/mixing'
# #WORK_DIR = 'drive/MyDrive/mixing'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.join(WORK_DIR, 'datasets'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default='/home/acg16879qo/openclip-cache',
        help='Directory for caching models from OpenCLIP'
    )

    parser.add_argument("--scaling-coef", default = 0.5)

    parser.add_argument("--base-path", type = str, help = "Base path where all experiments resulst are stored")
    
    parser.add_argument('--finetune-method', default = 'standard')
    parser.add_argument('--exp_id', type = int)
    
    parser.add_argument('--checkpoints', default = 'baseline', help ='If use new finetuned checkpoints, ths argument should be set to new')
    parser.add_argument('--alpha-feddyn', default = 0.5, help = 'Hyperparameter which controls the regularization term for finetuning by FedDyn')
    parser.add_argument('--save_what', default = 'finetuned', help = 'If save what == finetuned, only save fintuned checkpoints.')
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args

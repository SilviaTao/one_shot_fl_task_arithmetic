# Task Arithmetic Through The Lens Of One-Shot Federated Learning

This repository contains code for the paper [Task Arithmetic Through The Lens Of One-Shot Federated Learning](https://arxiv.org/abs/2411.18607). It is built upon the [repository](https://github.com/mlfoundations/task_vectors) for the paper [Editing Models With Task Arithmetic](https://arxiv.org/abs/2212.04089) and follows the same structure. 

## Dependencies
To run the code, please first install all dependencies and add directory to PYTHONPATH.

```
conda env create 
conda activate one_shot_fl_ta
cd one_shot_fl_task_arithmetic
export PYTHONPATH="$PYTHONPATH:$PWD"
```

## Fine-tuning
The module ```src/finetune.py``` is used to fine-tune models on downstream tasks. Below is an example of fine-tuning:

```
from src.finetune import finetune
from src.args import parse_arguments

args = parse_arguments()
args.train_dataset = 'MNISTVal' # For fine-tuning, append 'Val' to the dataset name
args.data_location = '/path/to/MNIST'
args.lr = 1e-5
args.epochs = 5
args.batch_size = 128
args.model = 'ViT-B-32'
arg.save = '/path/to/save/checkpoints'
finetune(args)
```

## Task Vectors
The process of creating and using task vectors follows the original [repository](https://github.com/mlfoundations/task_vectors). Please refer to it for more details. 

## Model Merging Methods
In this work, we implemented Task Arithmetic and its variants based on four Federated Learning algorithms: [FedNova](https://arxiv.org/abs/2007.07481), [FedGMA](https://arxiv.org/abs/2201.11986), [Median](https://arxiv.org/abs/1803.01498) and [CCLIP](https://arxiv.org/abs/2012.10333). The module ```src/ta_algorithms.py``` contains the implementation of all these algorithms, as well as the code for searching the best hyperparameters for each.

To run FedNova, provide a list of task vectors along with a corresponding list of the number of local steps used to train each task vector.

```
from src.ta_algorithms import fednova

task_vectors = [list of task vectors]
local_steps [list of ]
merged_model = fednova(task_vectors, )
```

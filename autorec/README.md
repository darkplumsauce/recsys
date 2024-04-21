# AutoRec (2015)

This PyTorch implementation follows the paper: [AutoRec: Autoencoders Meet Collaborative Filtering](https://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf)  (2015).

The goal is to train the user-item rating model using the auto-encoder architecture.

## Installation

Download the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/) into folder `data/ml-1m/*.dat`. Smaller version (100k) is also available. Then, set up the enviornment using command:

```bash
mamba env create -f environments.yml  # create conda env
conda activate recsys                 # activate env
```

## Running Instruction

```bash
python3 main.py # run the code
```

## Miscellaneous

Notable information missing from the paper includes:
- Batch size
- Number of epochs
- Certain optimizer parameters (learning rate, weight decay, etc.)

Functionality to be implemented:
- L2 regularization

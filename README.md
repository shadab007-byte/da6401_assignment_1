# DA6401 Assignment 1: Multi-Layer Perceptron for Image Classification

A fully configurable MLP built using **only NumPy** to classify MNIST and Fashion-MNIST.

## Links
- **W&B Report:** https://wandb.ai/iitm_assigment/da6401-assignment-1/reports/da6401_assignment1--VmlldzoxNjA0MjU1MA
- **GitHub:** https://github.com/shadab007-byte/da6401_assignment_1

## Project Structure
```
da6401_assignment_1/
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py           # sigmoid, tanh, relu, softmax
│   │   ├── neural_layer.py          # Dense layer with forward/backward + grad_W, grad_b
│   │   ├── neural_network.py        # Full MLP: forward(), backward(), train(), evaluate()
│   │   ├── objective_functions.py   # Cross-entropy and MSE losses
│   │   └── optimizers.py            # SGD, Momentum, NAG, RMSProp, Adam, Nadam
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Dataset loading, batching, one-hot encoding
│   │   └── metrics.py               # Accuracy, Precision, Recall, F1
│   ├── train.py                     # Training CLI script
│   └── inference.py                 # Inference CLI script
├── models/
│   ├── best_model.npy               # Best model weights 
│   └── best_config.json             # Best model configuration
├── notebooks/
│   └── da6401_colab.ipynb           # Full experiment notebook (all W&B questions)
├── README.md
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python src/train.py \
  -d fashion_mnist \
  -e 15 -b 128 \
  -o nadam -lr 0.001 \
  -nhl 3 -sz 128 \
  -a relu -l cross_entropy \
  -w_i xavier -wd 0.0001
```

## Inference
```bash
python src/inference.py \
  --model_path models/best_model.npy \
  --config_path models/best_config.json \
  --dataset mnist --split test
```

## Implemented Features
| Category       | Options |
|----------------|---------|
| Activations    | Sigmoid, Tanh, ReLU (hidden) + Softmax (output) |
| Loss Functions | Cross-Entropy, MSE |
| Optimizers     | SGD, Momentum, NAG, RMSProp, Adam, Nadam |
| Weight Init    | Random (Gaussian), Xavier (Glorot uniform) |
| Regularization | L2 weight decay |
| Metrics        | Accuracy, Precision, Recall, F1-score |

## Experiment Results

### MNIST (Best Configuration from Sweep)
| Optimizer | Activation | Layers | Hidden Size | LR | Batch | Val Accuracy |
|-----------|-----------|--------|-------------|-------|-------|-------------|
| Nadam | ReLU | 3 | 128 | 0.001 | 128 | 97.78% |
| Adam  | ReLU | 3 | 128 | 0.001 | 64  | 97.78% |
| Adam  | ReLU | 3 | 128 | 0.0005| 64  | 97.73% |

### Fashion-MNIST (Transfer from MNIST best configs)

| Configuration | Test Accuracy | Test F1 |
|--------------|--------------|---------|
| cfg1-nadam-relu-3L128 | 88.88% | 88.7% |
| cfg2-adam-relu-3L128  | 87.83% | 87.73% |
| cfg3-adam-relu-3L128-lr5e4 | 88.01% | 88.06% |

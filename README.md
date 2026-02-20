# DA6401 Assignment 1: Multi-Layer Perceptron for Image Classification

A fully configurable MLP built using **only NumPy** to classify MNIST and Fashion-MNIST.

## Links
- **W&B Report:** [Add your W&B report link here]
- **GitHub:** [This repository]

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
│   ├── best_model.npy               # Best model weights (by validation F1)
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
  -e 15 -b 64 \
  -o adam -lr 0.001 \
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

## Results
| Dataset       | Accuracy | F1    |
|---------------|----------|-------|
| MNIST         | ~98.2%   | ~98.2%|
| Fashion-MNIST | ~88.5%   | ~88.3%|

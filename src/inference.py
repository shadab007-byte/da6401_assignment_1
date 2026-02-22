import argparse
import sys
import os
import json
import numpy as np

# Allow imports from src/ when running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset, to_onehot, get_class_names
from utils.metrics import compute_metrics, get_confusion_matrix


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument('--model_path', type=str,
                        default='models/best_model.npy',
                        help='Path to saved model weights (.npy) — use relative path')

    #dataset: which dataset to evaluate on
    parser.add_argument('--dataset', type=str,
                        default=None,
                        choices=['mnist', 'fashion_mnist'],
                        help='Dataset to evaluate on (overrides config if given)')

    #batch_size: batch size for inference
    parser.add_argument('--batch_size', type=int,
                        default=256,
                        help='Batch size for inference')

    #hidden_layers: list of hidden layer sizes (e.g. --hidden_layers 128 128 128)
    parser.add_argument('--hidden_layers', type=int, nargs='+',
                        default=None,
                        help='List of hidden layer sizes, e.g. --hidden_layers 128 128 128')

    #num_neurons: neurons per layer when hidden_layers is not given
    parser.add_argument('--num_neurons', type=int,
                        default=128,
                        help='Number of neurons in each hidden layer')

    #activation: activation function
    parser.add_argument('--activation', type=str,
                        default='relu',
                        choices=['relu', 'sigmoid', 'tanh'],
                        help="Activation function ('relu', 'sigmoid', 'tanh')")

    #extra args needed to reconstruct model if no config file
    parser.add_argument('--config_path', type=str,
                        default='models/best_config.json',
                        help='Path to model config JSON (relative path)')
    parser.add_argument('--num_layers', type=int,
                        default=3,
                        help='Number of hidden layers (used if hidden_layers not given)')
    parser.add_argument('--weight_init', type=str,
                        default='xavier',
                        choices=['random', 'xavier'],
                        help='Weight initialization method used during training')
    parser.add_argument('--loss', type=str,
                        default='cross_entropy',
                        choices=['cross_entropy', 'mse'],
                        help='Loss function used during training')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0,
                        help='L2 weight decay used during training')
    parser.add_argument('--split', type=str,
                        default='test',
                        choices=['test', 'val', 'train'],
                        help='Which data split to evaluate on')

    return parser.parse_args()


def load_model(model_path):
   
    model_dir   = os.path.dirname(model_path) or 'models'
    config_path = os.path.join(model_dir, 'best_config.json')

    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"[INFO] Config loaded from: {config_path}")
    else:
        print(f"[WARN] Config not found at {config_path}. Using default architecture.")

    model_args = argparse.Namespace(
        num_layers   = len(config['hidden_sizes']) if config else 3,
        hidden_size  = config['hidden_sizes']      if config else [128, 128, 128],
        activation   = config.get('activation',   'relu'),
        weight_init  = config.get('weight_init',  'xavier'),
        loss         = config.get('loss',          'cross_entropy'),
        weight_decay = config.get('weight_decay',  0.0),
    )

    model = NeuralNetwork(cli_args=model_args)
    model.load_weights(model_path)

    return model, config


def evaluate_model(model, X_test, y_test):
    #convert integer labels to one-hot for loss computation
    y_onehot = to_onehot(y_test)

    #forward pass — returns logits (pre-softmax output of last layer)
    logits = model.forward(X_test)

    #compute loss using the model's own loss function
    loss, _ = model.loss_fn(logits, y_onehot)

    #predicted class = argmax of logits
    predictions = np.argmax(logits, axis=1)

    #compute all metrics via scikit-learn
    metrics = compute_metrics(y_test, predictions)

    #return exactly what the skeleton requires
    return {
        'logits':    logits,           # shape (N, 10)
        'loss':      float(loss),
        'accuracy':  metrics['accuracy'],
        'f1':        metrics['f1'],
        'precision': metrics['precision'],
        'recall':    metrics['recall'],
    }


def main():
    args = parse_arguments()

    #load model 
    model, config = load_model(args.model_path)

    #decide dataset
    dataset_name = args.dataset or config.get('dataset', 'fashion_mnist')

    #load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name)

    #select the correct split
    split_data = {
        'test':  (X_test,  y_test),
        'val':   (X_val,   y_val),
        'train': (X_train, y_train),
    }
    X_eval, y_eval = split_data[args.split]

    print(f"[INFO] Dataset : {dataset_name}")
    print(f"[INFO] Split   : {args.split} ({X_eval.shape[0]} samples)")

    #evaluate
    results = evaluate_model(model, X_eval, y_eval)

    #print results
    print("\n" + "=" * 50)
    print("         INFERENCE RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {results['accuracy']:.4f}  ({results['accuracy']*100:.2f}%)")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1-Score  : {results['f1']:.4f}")
    print(f"  Loss      : {results['loss']:.4f}")
    print("=" * 50)

    #per-class accuracy breakdown
    predictions = np.argmax(results['logits'], axis=1)
    class_names = get_class_names(dataset_name)

    print("\nPer-Class Accuracy:")
    for i, name in enumerate(class_names):
        mask      = (y_eval == i)
        class_acc = float(np.mean(predictions[mask] == y_eval[mask]))
        print(f"  Class {i:2d} ({name:15s}): {class_acc:.4f}")

    print("\nEvaluation complete!")

    #return dict
    return results


if __name__ == '__main__':
    main()
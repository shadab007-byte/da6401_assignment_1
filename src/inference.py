
import argparse
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset, to_onehot, get_class_names
from utils.metrics import compute_metrics


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on test set')
    parser.add_argument('--model_path', type=str, default='src/best_model.npy')
    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-nhl', '--num_layers', type=int, default=3)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 128, 128])
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier'])
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mse'])
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0001)
    parser.add_argument('-o', '--optimizer', type=str, default='sgd',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-wp', '--wandb_project', type=str, default='da6401-assignment-1')
    parser.add_argument('-we', '--wandb_entity', type=str, default='iitm_assigment')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'val', 'train'])
    return parser.parse_args()


def load_model(model_path):
    """Load trained model from disk. Returns raw weights dict."""
    data = np.load(model_path, allow_pickle=True)
    data = data.tolist()
    return data


def evaluate_model(model, X_test, y_test):
    y_onehot    = to_onehot(y_test)
    logits      = model.forward(X_test)
    loss, _     = model.loss_fn(logits, y_onehot)
    predictions = np.argmax(logits, axis=1)
    metrics     = compute_metrics(y_test, predictions)
    return {
        'logits':    logits,
        'loss':      float(loss),
        'accuracy':  metrics['accuracy'],
        'f1':        metrics['f1'],
        'precision': metrics['precision'],
        'recall':    metrics['recall'],
    }


def main():
    args = parse_arguments()

    
    model = NeuralNetwork(cli_args=args)
    weights = load_model(args.model_path)
    model.set_weights(weights)
    best_weights = model.get_weights()
    np.save(args.model_path, best_weights)

    dataset_name = args.dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(dataset_name)

    split_data = {
        'test':  (X_test,  y_test),
        'val':   (X_val,   y_val),
        'train': (X_train, y_train),
    }
    X_eval, y_eval = split_data[args.split]

    print(f"[INFO] Dataset : {dataset_name}")
    print(f"[INFO] Split   : {args.split} ({X_eval.shape[0]} samples)")

    results = evaluate_model(model, X_eval, y_eval)

    print("\n" + "=" * 50)
    print("         INFERENCE RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1-Score  : {results['f1']:.4f}")
    print(f"  Loss      : {results['loss']:.4f}")
    print("=" * 50)

    predictions = np.argmax(results['logits'], axis=1)
    class_names = get_class_names(dataset_name)
    print("\nPer-Class Accuracy:")
    for i, name in enumerate(class_names):
        mask      = (y_eval == i)
        class_acc = float(np.mean(predictions[mask] == y_eval[mask]))
        print(f"  Class {i:2d} ({name:15s}): {class_acc:.4f}")

    print("\nEvaluation complete!")
    return results


if __name__ == '__main__':
    main()

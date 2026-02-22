import argparse
import sys
import os
import json
import numpy as np

# Allow imports from src/ when running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wandb
from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from utils.data_loader import load_dataset, to_onehot, get_batches
from utils.metrics import compute_metrics


def parse_arguments():
    """
    - dataset       : 'mnist' or 'fashion_mnist'            [-d]
    - epochs        : Number of training epochs             [-e]
    - batch_size    : Mini-batch size                       [-b]
    - learning_rate : Learning rate for optimizer           [-lr]
    - optimizer     : sgd/momentum/nag/rmsprop/adam/nadam   [-o]
    - num_layers    : Number of hidden layers               [-nhl]
    - hidden_size   : Number of neurons per hidden layer    [-sz]
    - activation    : relu / sigmoid / tanh                 [-a]
    - loss          : cross_entropy / mse                   [-l]
    - weight_init   : random or xavier                      [-w_i]
    - weight_decay  : L2 regularization coefficient        [-wd]
    - wandb_project : W&B da6401-assigment-1
    - model_save_path : Path to save trained model (relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    #dataset
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',
                        choices=['mnist', 'fashion_mnist'],
                        help="Dataset to use: 'mnist' or 'fashion_mnist'")

    #training hyperparameters
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Mini-batch size for SGD')

    #loss function
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mse'],
                        help='Loss function: cross_entropy or mse')

    #optimizer
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimizer to use')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0,
                        help='L2 weight decay (regularization) coefficient')

    #optimizer momentum/beta parameters
    parser.add_argument('--beta', type=float, default=0.9,
                        help='Momentum beta (also beta1 for Adam/Nadam)')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Beta2 for Adam/Nadam second moment')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Epsilon for numerical stability in Adam/RMSProp')

    #network architecture
    parser.add_argument('-nhl', '--num_layers', type=int, default=3,
                        help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128],
                        help='Number of neurons in each hidden layer. '
                             'Give one value (same for all) or one per layer.')
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'],
                        help="Activation function ('relu', 'sigmoid', 'tanh')")
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier'],
                        help='Weight initialization method')

    #W&B settings
    parser.add_argument('--wandb_project', type=str, default='da6401-assignment-1',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity. Leave blank for default.')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Custom name for this W&B run')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable W&B logging (useful for quick local tests)')

    #save paths
    parser.add_argument('--model_save_path', type=str,
                        default='models/best_model.npy',
                        help='Relative path to save best model weights (.npy)')
    parser.add_argument('--config_save_path', type=str,
                        default='models/best_config.json',
                        help='Relative path to save best model config (.json)')

    #extra W&B logging flags (used for specific experiments)
    parser.add_argument('--log_grad_norms', action='store_true',
                        help='Log per-layer gradient norms to W&B')
    parser.add_argument('--log_activations', action='store_true',
                        help='Log activation statistics to W&B')

    return parser.parse_args()


def main():
    args = parse_arguments()

    #load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args.dataset)
    y_train_oh = to_onehot(y_train)   # one-hot for training
    y_val_oh   = to_onehot(y_val)
    y_test_oh  = to_onehot(y_test)

    #build model
    model = NeuralNetwork(cli_args=args)

    #build optimizer
    #pass only the kwargs each optimizer actually uses
    opt_kwargs = dict(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    if args.optimizer in ['momentum', 'nag']:
        opt_kwargs['beta'] = args.beta
    elif args.optimizer == 'rmsprop':
        opt_kwargs['beta'] = args.beta
        opt_kwargs['eps']  = args.eps
    elif args.optimizer in ['adam', 'nadam']:
        opt_kwargs['beta1'] = args.beta
        opt_kwargs['beta2'] = args.beta2
        opt_kwargs['eps']   = args.eps

    model.optimizer = get_optimizer(args.optimizer, **opt_kwargs)
    use_nag = model.optimizer.is_nag   # flag: True only for NAG

    #W&B init
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    #training loop
    best_val_f1 = -1.0

    for epoch in range(1, args.epochs + 1):

        epoch_losses = []

        for X_b, y_b in get_batches(X_train, y_train_oh, args.batch_size, shuffle=True):

            #NAG: shift weights to lookahead position BEFORE forward pass
            if use_nag:
                model.optimizer.apply_lookahead(model.layers)

            #forward pass (at lookahead position for NAG current W for others)
            logits = model.forward(X_b)

            #backward pass: computes grad_W and grad_b for every layer
            #for NAG: gradient is at the lookahead position
            model.backward(y_b, logits)

            #update weights
            #for NAG: restores W_original and applies v correctly
            model.update_weights()

            epoch_losses.append(model._last_loss)

        #Per-epoch metrics
        avg_train_loss = float(np.mean(epoch_losses))
        train_result   = model.evaluate(X_train, y_train_oh)
        val_result     = model.evaluate(X_val,   y_val_oh)
        val_metrics    = compute_metrics(y_val, val_result['predictions'])

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"TrainLoss: {avg_train_loss:.4f} | "
            f"TrainAcc: {train_result['accuracy']:.4f} | "
            f"ValLoss: {val_result['loss']:.4f} | "
            f"ValAcc: {val_result['accuracy']:.4f} | "
            f"ValF1: {val_metrics['f1']:.4f}"
        )

        #W&B logging
        log_dict = {
            'epoch':          epoch,
            'train_loss':     avg_train_loss,
            'train_accuracy': train_result['accuracy'],
            'val_loss':       val_result['loss'],
            'val_accuracy':   val_result['accuracy'],
            'val_f1':         val_metrics['f1'],
            'val_precision':  val_metrics['precision'],
            'val_recall':     val_metrics['recall'],
        }

        #gradient norms per layer (vanishing gradient)
        if args.log_grad_norms:
            for name, norm in model.get_gradient_norms():
                log_dict[name] = norm

        #activation stats per layer (dead neuron)
        if args.log_activations:
            log_dict.update(model.get_activation_stats())

        if use_wandb:
            wandb.log(log_dict)

        #save best model (by validation F1 score)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            model.save_weights(args.model_save_path)

            save_config = {
                'dataset':       args.dataset,
                'hidden_sizes':  model.hidden_sizes,
                'activation':    args.activation,
                'weight_init':   args.weight_init,
                'loss':          args.loss,
                'weight_decay':  args.weight_decay,
                'optimizer':     args.optimizer,
                'learning_rate': args.learning_rate,
                'num_layers':    args.num_layers,
            }
            os.makedirs(os.path.dirname(args.config_save_path) or '.', exist_ok=True)
            with open(args.config_save_path, 'w') as f:
                json.dump(save_config, f, indent=2)

    #final test evaluation (using best saved weights)
    model.load_weights(args.model_save_path)
    test_result  = model.evaluate(X_test, y_test_oh)
    test_metrics = compute_metrics(y_test, test_result['predictions'])

    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS")
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}")
    print(f"  F1-Score  : {test_metrics['f1']:.4f}")
    print("=" * 50)

    if use_wandb:
        wandb.log({
            'test_accuracy':  test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall':    test_metrics['recall'],
            'test_f1':        test_metrics['f1'],
        })
        wandb.finish()

    print("Training complete!")


if __name__ == '__main__':
    main()
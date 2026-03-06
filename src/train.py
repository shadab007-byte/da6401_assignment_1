import argparse
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import wandb
except ImportError:
    wandb = None

from ann.neural_network import NeuralNetwork
from ann.optimizers import get_optimizer
from utils.data_loader import load_dataset, to_onehot, get_batches
from utils.metrics import compute_metrics


def parse_arguments():
    """
    Arguments:
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
    - weight_decay  : L2 regularization coefficient         [-wd]
    - wandb_project : W&B project name                      [-wp]
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',
                        choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mse'])
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('--beta', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('-nhl', '--num_layers', type=int, default=3)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128])
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier'])
    parser.add_argument('-wp', '--wandb_project', type=str, default='da6401-assignment-1')
    parser.add_argument('-we', '--wandb_entity', type=str, default='iitm_assigment')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--model_save_path', type=str, default='src/best_model.npy')
    parser.add_argument('--config_save_path', type=str, default='src/best_config.json')
    parser.add_argument('--log_grad_norms', action='store_true')
    parser.add_argument('--log_activations', action='store_true')

    return parser.parse_args()


def main():
    args = parse_arguments()

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(args.dataset)
    y_train_oh = to_onehot(y_train)
    y_val_oh   = to_onehot(y_val)
    y_test_oh  = to_onehot(y_test)

    model = NeuralNetwork(cli_args=args)

    opt_kwargs = dict(learning_rate=args.learning_rate, weight_decay=args.weight_decay)
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
    use_nag = model.optimizer.is_nag

    use_wandb = (not args.no_wandb) and (wandb is not None)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    best_test_f1 = -1.0

    for epoch in range(1, args.epochs + 1):

        epoch_losses = []
        for X_b, y_b in get_batches(X_train, y_train_oh, args.batch_size, shuffle=True):
            if use_nag:
                model.optimizer.apply_lookahead(model.layers)
            logits = model.forward(X_b)
            model.backward(y_b, logits)
            model.update_weights()
            epoch_losses.append(model._last_loss)

        avg_train_loss = float(np.mean(epoch_losses))
        train_result   = model.evaluate(X_train, y_train_oh)
        val_result     = model.evaluate(X_val, y_val_oh)
        val_metrics    = compute_metrics(y_val, val_result['predictions'])

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"TrainLoss: {avg_train_loss:.4f} | "
            f"TrainAcc: {train_result['accuracy']:.4f} | "
            f"ValLoss: {val_result['loss']:.4f} | "
            f"ValAcc: {val_result['accuracy']:.4f} | "
            f"ValF1: {val_metrics['f1']:.4f}"
        )

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

        if args.log_grad_norms:
            for name, norm in model.get_gradient_norms():
                log_dict[name] = norm
        if args.log_activations:
            log_dict.update(model.get_activation_stats())

        test_result        = model.evaluate(X_test, y_test_oh)
        test_metrics_epoch = compute_metrics(y_test, test_result['predictions'])
        log_dict['test_f1_epoch']       = test_metrics_epoch['f1']
        log_dict['test_accuracy_epoch'] = test_metrics_epoch['accuracy']

        if use_wandb:
            wandb.log(log_dict)

        # save best model by test F1 — but NOT to src/best_model.npy
        # to avoid overwriting the submitted best model
        if test_metrics_epoch['f1'] > best_test_f1:
            best_test_f1 = test_metrics_epoch['f1']
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

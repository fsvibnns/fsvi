import os
import json
import argparse
import torch
import numpy as np

print("NOTE: To run baselines, clone baseline repos and set appropriate working directories.")

# We import the following radial_layers so that we can initialize objects of the correct class using the config files
import radial_layers.loss as module_loss

import baselines.uncertainty_quantification.radial_bnn.model.metric_mod as metric
import baselines.uncertainty_quantification.radial_bnn.model.model_mod as module_model

from fsvi.utils import datasets

from trainer import Trainer  # object to manage training and validation loops
from radial_utils.util import manage_seed  # helper functions

def main(config, resume, task):
    """
    Completes single optimization run.
    :param config: Dictionary from configuration file. See example in configs/
    :param resume: Path to checkpoint to resume from.
    :return: monitor_best, monitor last, monitor_best_se (the best metric measured, the final metric measured,
    the standard error of the best metric measured)
    """

    seed = manage_seed("random")  # You may want to log this with whatever tool you prefer

    # Setup data_loader instances
    if task == "fashionmnist":
        data_training = "fashionmnist"
        data_ood = "mnist"
    elif task == "cifar10":
        data_training = "cifar10"
        data_ood = "svhn"

    model_type = "map_cnn"
    batch_size = 200
    val_frac = 0.0

    (   trainloader,
        x_train_permuted,
        y_train_permuted,
        x_test,
        y_test,
        x_ood,
        y_ood,
        input_shape,
        input_dim,
        output_dim,
        n_train,
        n_batches,
    ) = datasets.load_data(
        model_type=model_type,
        data_training=data_training,
        data_ood=data_ood,
        batch_size=batch_size,
        seed=seed,
        val_frac=val_frac,
    )

    # Build models
    model = getattr(module_model, config["arch"]["type"])(**config["arch"]["args"])
    model.summary()

    # Set the loss
    num_batches = len(trainloader)
    loss = getattr(module_loss, config["loss"]["type"])(**config["loss"]["args"])
    loss.set_num_batches(num_batches)

    if hasattr(loss, "set_model"):
        # The ELBO loss needs to know the batch size to correctly balance factors
        loss.set_model(model, config)

    # build optimizer.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(torch.optim, config["optimizer"]["type"])(trainable_params,
                                                                  **config["optimizer"]["args"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    x_test_torch = torch.from_numpy(x_test.transpose([0, 3, 1, 2])).to(device)
    test_target_torch = torch.from_numpy(y_test.argmax(-1)).to(device)
    x_ood_torch = torch.from_numpy(x_ood.transpose([0, 3, 1, 2])).to(device)

    epochs = 100
    variational_samples = 5

    for epoch in range(epochs):
        model.train()

        for batch_idx, (data, target) in enumerate(trainloader):
            optimizer.zero_grad()
            data = data.to(device)
            data = data.unsqueeze(1)
            target = target.to(device)

            output = model(data)

            nll_loss, kl_term = loss.compute_loss(output, target)
            batch_loss = nll_loss + kl_term
            batch_loss.backward()
            optimizer.step()

        model.eval()
        total_test_nll = 0.0
        total_test_entropy = 0.0
        total_test_accuracy = 0.0

        total_test_preds_list = []

        with torch.no_grad():
            x_test = x_test_torch.unsqueeze(1).expand((-1, variational_samples, -1, -1, -1))
            for batch_idx in range(10):
                output = model(x_test[batch_idx * 1000: (batch_idx + 1) * 1000, :])
                test_target = test_target_torch[batch_idx * 1000: (batch_idx + 1) * 1000].detach().cpu().numpy()
                test_output = output.detach().cpu().numpy()

                total_test_preds_list.append(test_output)

                test_accuracy = metric.accuracy(test_output, test_target)
                total_test_accuracy += test_accuracy

                test_nll = metric._nll(test_output, test_target)
                total_test_nll += test_nll

                test_entropy = metric.predictive_entropy(np.exp(test_output).mean(1)).mean()
                total_test_entropy += test_entropy

            test_accuracy_mean = total_test_accuracy / 10.
            test_nll_mean = total_test_nll / 10.
            test_entropy_mean = total_test_entropy / 10.

            total_test_preds = np.array(total_test_preds_list)

            print("Epoch: ", epoch)
            print("Test Accuracy: ", test_accuracy_mean)
            print("Test NLL: ", test_nll_mean)
            print("Test Entropy: ", test_entropy_mean)

            total_ood_entropy = 0.0
            total_ood_preds_list = []

            x_ood = x_ood_torch.unsqueeze(1).expand((-1, variational_samples, -1, -1, -1))
            for batch_idx in range(10):
                output = model(x_ood[batch_idx * 1000: (batch_idx + 1) * 1000, :])

                ood_output = output.detach().cpu().numpy()

                total_ood_preds_list.append(ood_output)

                ood_entropy = metric.predictive_entropy(np.exp(ood_output).mean(1)).mean()
                total_ood_entropy += ood_entropy

            total_ood_preds = np.array(total_ood_preds_list)
            total_ood_preds = total_ood_preds.reshape([-1, variational_samples, 10])
            total_test_preds = total_test_preds.reshape([-1, variational_samples, 10])

            auroc_entropy = metric.auroc(total_test_preds, total_ood_preds, score="entropy")

            ood_entropy_mean = total_ood_entropy / 10

            print("OOD Entropy: ", ood_entropy_mean)
            print("AUROC Entropy: ", auroc_entropy)

    torch.save(model.state_dict(), f"../saved_models/radial_bnn_{task}_{seed}")
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('-t', '--task', default=None, type=str,
                        help='Task')
    args = parser.parse_args()

    if args.config:
        # load config file if one is provided
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
        # At the moment, preferred default behaviour is to fail here. Comment out assertion if you want
        # to use 'config.json' as a default.
        config = json.load(open('config.json'))
        path = os.path.join(config['trainer']['save_dir'], config['name'])

    print(torch.cuda.is_available())
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(config, args.resume, args.task)




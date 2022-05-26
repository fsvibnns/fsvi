import argparse
import os, sys
import time
import tabulate

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
# from pytorch_model_summary import summary

print("NOTE: To run baselines, clone baseline repos and set appropriate working directories.")

from swag import data, utils, losses
from swag.posteriors import SWAG

from baselines.uncertainty_quantification.swag import models


eps = 1e-10

from sklearn.metrics import roc_auc_score, roc_curve


def predictive_entropy(predicted_labels):
    entropy = -((predicted_labels + eps) * np.log(predicted_labels + eps)).sum(-1)
    return entropy


def auroc(predicted_labels_test, predicted_labels_ood, score):
    ood_size = predicted_labels_ood.shape[1]
    test_size = predicted_labels_test.shape[1]
    anomaly_targets = np.concatenate((np.zeros(test_size), np.ones(ood_size)))
    if score == "entropy":
        entropy_test = predictive_entropy(predicted_labels_test.mean(0))
        entropy_ood = predictive_entropy(predicted_labels_ood.mean(0))
        scores = np.concatenate((entropy_test, entropy_ood))
    if score == "expected entropy":
        entropy_test = predictive_entropy(predicted_labels_test).mean(0)
        entropy_ood = predictive_entropy(predicted_labels_ood).mean(0)
        scores = np.concatenate((entropy_test, entropy_ood))
    elif score == "mutual information":
        mutual_information_test = np.mean(
            np.mean(
                np.square(
                    predicted_labels_test - predicted_labels_test.mean(0),
                    dtype=np.float64,
                ),
                0,
                dtype=np.float64,
            ),
            -1,
            dtype=np.float64,
        )
        mutual_information_ood = np.mean(
            np.mean(
                np.square(
                    predicted_labels_ood - predicted_labels_ood.mean(0),
                    dtype=np.float64,
                ),
                0,
                dtype=np.float64,
            ),
            -1,
            dtype=np.float64,
        )
        scores = np.concatenate((mutual_information_test, mutual_information_ood))
    else:
        NotImplementedError
    fpr, tpr, _ = roc_curve(anomaly_targets, scores)
    auroc_score = roc_auc_score(anomaly_targets, scores)
    return auroc_score


parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument(
    "--dir",
    type=str,
    default=None,
    required=True,
    help="training directory (default: None)",
)

parser.add_argument(
    "--data_training", type=str, default="CIFAR10", help="dataset name (default: CIFAR10)"
)
parser.add_argument(
    "--data_ood",
    nargs="+",
    default=["not_specified"],
    help="Out-of-distribution dataset used (default: [not_specified])",
)
parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    required=True,
    metavar="PATH",
    help="path to datasets location (default: None)",
)
parser.add_argument(
    "--use_test",
    dest="use_test",
    action="store_true",
    help="use test dataset instead of validation (default: False)",
)
parser.add_argument("--split_classes", type=int, default=None)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    metavar="MODEL",
    help="model name (default: None)",
)

parser.add_argument(
    "--resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to resume training from (default: None)",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 200)",
)
parser.add_argument(
    "--save_freq",
    type=int,
    default=25,
    metavar="N",
    help="save frequency (default: 25)",
)
parser.add_argument(
    "--eval_freq",
    type=int,
    default=1,
    metavar="N",
    help="evaluation frequency (default: 1)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.01,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="SGD momentum (default: 0.9)",
)
parser.add_argument(
    "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
)

parser.add_argument("--swa", action="store_true", help="swa usage flag (default: off)")
parser.add_argument(
    "--swa_start",
    type=float,
    default=161,
    metavar="N",
    help="SWA start epoch number (default: 161)",
)
parser.add_argument(
    "--swa_lr", type=float, default=0.02, metavar="LR", help="SWA LR (default: 0.02)"
)
parser.add_argument(
    "--swa_c_epochs",
    type=int,
    default=1,
    metavar="N",
    help="SWA model collection frequency/cycle length in epochs (default: 1)",
)
parser.add_argument("--cov_mat", action="store_true", help="save sample covariance")
parser.add_argument(
    "--max_num_models",
    type=int,
    default=20,
    help="maximum number of SWAG models to save",
)

parser.add_argument(
    "--swa_resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to restor SWA from (default: None)",
)
parser.add_argument(
    "--loss",
    type=str,
    default="CE",
    help="loss to use for training model (default: Cross-entropy)",
)
# ADDED
parser.add_argument(
    "--optimizer",
    type=str,
    default="sgd",
    help="Optimzer (adam or sgd)",
)

parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument("--no_schedule", action="store_true", help="store schedule")

args = parser.parse_args()

args.device = None

use_cuda = torch.cuda.is_available()

if use_cuda:
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

print("Preparing directory %s" % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, "command.sh"), "w") as f:
    f.write(" ".join(sys.argv))
    f.write("\n")

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)

if use_cuda:
    torch.cuda.manual_seed(args.seed)

print("Using model %s" % args.model)
model_cfg = getattr(models, args.model)

# print("Loading dataset %s from %s" % (args.dataset, args.data_path))
# loaders_old, num_classes_old = data.loaders(
#     args.dataset,
#     args.data_path,
#     args.batch_size,
#     args.num_workers,
#     model_cfg.transform_train,
#     model_cfg.transform_test,
#     use_validation=not args.use_test,
#     split_classes=args.split_classes,
# )

from fsvi.utils import datasets

data_training = args.data_training
data_ood = args.data_ood

model_type = "map_cnn"
batch_size = args.batch_size
seed = args.seed
val_frac = 0.0

(
    trainloader,
    testloader,
    x_train_permuted,
    y_train_permuted,
    x_test,
    y_test,
    x_ood,
    y_ood,
    input_shape,
    input_dim,
    num_classes,
    n_train,
    n_batches,
) = datasets.load_data(
    model_type=model_type,
    data_training=data_training,
    data_ood=data_ood,
    batch_size=batch_size,
    seed=seed,
    val_frac=val_frac,
    return_testloader=True,
)

x_test = x_test.transpose(0, 3, 1, 2)
x_ood = [x_ood[0].transpose(0, 3, 1, 2), x_ood[1].transpose(0, 3, 1, 2)]

loaders = {}
loaders["train"] = trainloader
loaders["test"] = testloader

print("Preparing model")
print(*model_cfg.args)
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)


if args.cov_mat:
    args.no_cov_mat = False
else:
    args.no_cov_mat = True
if args.swa:
    print("SWAG training")
    swag_model = SWAG(
        model_cfg.base,
        no_cov_mat=args.no_cov_mat,
        max_num_models=args.max_num_models,
        *model_cfg.args,
        num_classes=num_classes,
        **model_cfg.kwargs,
    )
    swag_model.to(args.device)
else:
    print(f"Training with {args.optimizer} (without SWAG)")


def schedule(epoch):
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


# use a slightly modified loss function that allows input of model
if args.loss == "CE":
    criterion = losses.cross_entropy
    # criterion = F.cross_entropy
elif args.loss == "adv_CE":
    criterion = losses.adversarial_cross_entropy

if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_init,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
elif args.optimizer == "adam":
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr_init, weight_decay=args.wd
    )
else:
    NotImplementedError("Optimizer not specified.")

start_epoch = 0
if args.resume is not None:
    print("Resume training from %s" % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

if args.swa and args.swa_resume is not None:
    checkpoint = torch.load(args.swa_resume)
    swag_model = SWAG(
        model_cfg.base,
        no_cov_mat=args.no_cov_mat,
        max_num_models=args.max_num_models,
        loading=True,
        *model_cfg.args,
        num_classes=num_classes,
        **model_cfg.kwargs,
    )
    swag_model.to(args.device)
    swag_model.load_state_dict(checkpoint["state_dict"])

columns = [
    "ep",
    "lr",
    "tr_loss",
    "tr_acc",
    "te_loss",
    "te_acc",
    "te_entr",
    "ood_entr",
    "auroc",
    "time",
    "mem_usage",
]
if args.swa:
    columns = columns[:-2] + ["swa_te_loss", "swa_te_acc"] + columns[-2:]
    swag_res = {"loss": None, "accuracy": None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict(),
)

sgd_ens_preds = None
sgd_targets = None
n_ensembled = 0.0

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    if not args.no_schedule:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    if (args.swa and (epoch + 1) > args.swa_start) and args.cov_mat:
        train_res = utils.train_epoch(
            loaders["train"], model, criterion, optimizer, cuda=use_cuda
        )
    else:
        train_res = utils.train_epoch(
            loaders["train"], model, criterion, optimizer, cuda=use_cuda
        )

    test_res = utils.eval(loaders["test"], model, criterion, cuda=use_cuda)

    if (
        args.swa
        and (epoch + 1) > args.swa_start
        and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0
    ):
        # sgd_preds, sgd_targets = radial_utils.predictions(loaders["test"], model)
        sgd_res = utils.predict(loaders["test"], model)
        sgd_preds = sgd_res["predictions"]
        sgd_targets = sgd_res["targets"]
        print("updating sgd_ens")
        if sgd_ens_preds is None:
            sgd_ens_preds = sgd_preds.copy()
        else:
            # TODO: rewrite in a numerically stable way
            sgd_ens_preds = sgd_ens_preds * n_ensembled / (
                n_ensembled + 1
            ) + sgd_preds / (n_ensembled + 1)
        n_ensembled += 1
        swag_model.collect_model(model)
        if (
            epoch == 0
            or epoch % args.eval_freq == args.eval_freq - 1
            or epoch == args.epochs - 1
        ):
            swag_model.sample(0.0)
            utils.bn_update(loaders["train"], swag_model)
            swag_res = utils.eval(loaders["test"], swag_model, criterion)
        else:
            swag_res = {"loss": None, "accuracy": None}

    # if (epoch + 1) % args.save_freq == 0:
    #     radial_utils.save_checkpoint(
    #         args.dir,
    #         epoch + 1,
    #         state_dict=model.state_dict(),
    #         optimizer=optimizer.state_dict(),
    #     )
    #     if args.swa:
    #         radial_utils.save_checkpoint(
    #             args.dir, epoch + 1, name="swag", state_dict=swag_model.state_dict()
    #         )

    time_ep = time.time() - time_ep

    if use_cuda:
        memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)

    # ADDED FOR OOD
    preds_ood = []

    if (
        args.swa
        and (epoch + 1) > args.swa_start
        and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0
    ):
        swag_model.eval()
        with torch.no_grad():
            input = torch.Tensor(x_test).cuda()
            preds_test = swag_model(input)
            for x_ood_tp in x_ood:
                input = torch.Tensor(x_ood_tp[:10000]).cuda()
                preds_ood.append(swag_model(input))
    else:
        model.eval()
        with torch.no_grad():
            input = torch.Tensor(x_test).cuda()
            preds_test = model(input)
            for x_ood_tp in x_ood:
                input = torch.Tensor(x_ood_tp[:10000]).cuda()
                preds_ood.append(model(input))

    preds_test = F.softmax(preds_test, 1).cpu().detach().numpy()
    preds_ood = [F.softmax(preds, 1).cpu().detach().numpy() for preds in preds_ood]
    # preds_test = np.exp(preds_test.cpu().detach().numpy())
    # preds_ood = np.exp(preds_ood.cpu().detach().numpy())
    entropy_test = predictive_entropy(preds_test).mean()
    entropy_ood = [np.round(predictive_entropy(preds).mean(), 3) for preds in preds_ood]
    preds_test = np.reshape(preds_test, (1, preds_test.shape[0], preds_test.shape[1]))
    preds_ood = [np.reshape(preds, (1, preds.shape[0], preds.shape[1])) for preds in preds_ood]
    auroc_score = [np.round(auroc(preds_test, preds, "entropy"), 3) for preds in preds_ood]

    values = [
        epoch + 1,
        lr,
        train_res["loss"],
        train_res["accuracy"],
        test_res["loss"],
        test_res["accuracy"],
        entropy_test,
        entropy_ood,
        auroc_score,
        time_ep,
        memory_usage,
    ]
    if args.swa:
        values = values[:-2] + [swag_res["loss"], swag_res["accuracy"]] + values[-2:]
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)

# if args.epochs % args.save_freq != 0:
#     radial_utils.save_checkpoint(
#         args.dir,
#         args.epochs,
#         state_dict=model.state_dict(),
#         optimizer=optimizer.state_dict(),
#     )
#     if args.swa and args.epochs > args.swa_start:
#         radial_utils.save_checkpoint(
#             args.dir, args.epochs, name="swag", state_dict=swag_model.state_dict()
#         )
#
# if args.swa:
#     np.savez(
#         os.path.join(args.dir, "sgd_ens_preds.npz"),
#         predictions=sgd_ens_preds,
#         targets=sgd_targets,
#     )
if args.swa:
    torch.save(swag_model.state_dict(), f"../saved_models/swag_{args.data_training}_{args.seed}")
    np.save(f"../saved_models/swag_{args.data_training}_{args.data_ood}_{args.seed}_test_labels", preds_test)
    np.save(f"../saved_models/swag_{args.data_training}_{args.data_ood[0]}_{args.seed}_ood_labels", preds_ood[0])
    np.save(f"../saved_models/swag_{args.data_training}_{args.data_ood[1]}_{args.seed}_ood_labels", preds_ood[1])


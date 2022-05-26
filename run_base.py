import argparse
import getpass
import csv
import json
import os
import pdb
import sys
from typing import Dict
from copy import copy

# MAKE DETERMINISTIC
os.environ['PYTHONHASHSEED']=str(0)  # TODO: check how this affects determinism -- keep commented out
os.environ["TF_DETERMINISTIC_OPS"] = "1"  # TODO: check how this affects determinism -- keep commented out
deterministic_gpu = True
if deterministic_gpu:
    print(f'Making GPU operations deterministic by setting '
          f'os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_reductions""'
          f'and '
          f'os.environ["TF_CUDNN_DETERMINISTIC"] = "1"')
    # os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_reductions " + os.environ["XLA_FLAGS"]
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# if getpass.getuser() == "ANON":
#     os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda"
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tf_cpu_only = False  # TODO: check how this affects determinism -- keep set to False
if tf_cpu_only:
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], "GPU")
    print('WARNING: TensorFlow is set to only use CPU.')

# FOR DEBUGGING
# from jax import config
# config.update('jax_disable_jit', True)
# config.update('jax_platform_name', 'cpu')
# config.update("jax_enable_x64", True)

root_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, root_folder)
sys.path.insert(0, os.path.join(root_folder, "function_space_variational_inference"))

from jax.lib import xla_bridge

try:
    from fsvi.trainers.classification_ood import classification_ood
except:
    print("WARNING: Classification OOD training could not be loaded")

from fsvi.general_utils.log import create_logdir, set_up_logging

try:
    from fsvi.trainers.regression import regression
    from fsvi.trainers.two_moons import classification_two_moons
except:
    print("WARNING: One or more trainers could not be loaded")
try:
    from fsvi.trainers.uci import uci
except:
    print("WARNING: UCI trainer could not be loaded")
try:
    from fsvi.trainers.offline_rl import offline_rl
except:
    print("WARNING: Offline RL trainer could not be loaded")
try:
    from fsvi.trainers.offline_rl import offline_rl_door_eval
except:
    print("WARNING: Offline RL evaluator could not be loaded")
try:
    from fsvi.trainers.causal_inference import causal_inference
except:
    print("WARNING: Causal inference trainer could not be loaded")
try:
    from src_cl.trainers.continual_learning import continual_learning
except:
    print("WARNING: Continual learning trainer could not be loaded")


def main(kwargs):
    # Select trainer
    if (
        "mnist_" in kwargs["task"]
        or "fashionmnist_" in kwargs["task"]
        or "cifar10_" in kwargs["task"]
    ):
        trainer = classification_ood
    elif (
        "snelson" in kwargs["task"]
        or "solar" in kwargs["task"]
        or "oat1d" in kwargs["task"]
        or "subspace_inference" in kwargs["task"]
        or "uci" in kwargs["task"]
    ):
        trainer = regression
    elif "ihdp" in kwargs["task"]:
        trainer = causal_inference
    elif "two_moons" in kwargs["task"]:
        trainer = classification_two_moons
    elif "offline_rl" in kwargs["task"]:
        if "eval" in kwargs["task"]:
            trainer = offline_rl_door_eval
        else:
            trainer = offline_rl
    elif "continual_learning" in kwargs["task"]:
        trainer = continual_learning
    else:
        NotImplementedError("Experiment specification not recognized")

    # PASS VARIABLES TO TRAINER
    trainer(**kwargs)

    if kwargs["save"] and not kwargs["resume_training"]:
        save_path = kwargs["save_path"]
        print(
            f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n"
        )
        os.rename(save_path, f"{save_path}__complete")

    print("\n------------------- DONE -------------------\n")


def add_base_args(parser):
    parser.add_argument(
        "--data_training",
        type=str,
        default="not_specified",
        help="Training and in-distribution dataset used (default: not_specified)\n"
             "Examples: 'continual_learning_pmnist', 'continual_learning_smnist', "
             "'continual_learning_sfashionmnist'",
    )

    parser.add_argument(
        "--data_ood",
        nargs="+",
        default=["not_specified"],
        help="Out-of-distribution dataset used (default: [not_specified])",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="not_specified",
        help="Model used (default: not_specified). Example: 'fsvi_mlp', 'mfvi_cnn'",
    )

    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer used (default: adam)"
    )

    parser.add_argument(
        "--optimizer_var", type=str, default="not_specified", help="Optimizer used for variance paramters (default: not_specified)"
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        help="Momentum in SGD",
    )

    parser.add_argument(
        "--momentum_var",
        type=float,
        default=0.0,
        help="Momentum in SGD for variance parameters",
    )

    parser.add_argument(
        "--schedule",
        type=str,
        default="not_specified",
        help="Learning rate schedule type (default: not_specified)",
    )

    parser.add_argument(
        "--architecture",
        type=str,
        default="not_specified",
        help="Architecture of NN (default: not_specified)",
    )

    parser.add_argument(
        "--activation",
        type=str,
        default="not_specified",
        help="Activation function used in NN (default: not_specified)",
    )

    parser.add_argument(
        "--prior_mean", type=str, default="0", help="Prior mean function (default: 0)"
    )

    parser.add_argument(
        "--prior_cov", type=str, default="0", help="Prior cov function (default: 0)"
    )

    parser.add_argument(
        "--prior_covs",
        nargs="+",
        default=[0.0],
        type=float,
        help="prior_covs used (default: [0.0])",
    )

    parser.add_argument(
        "--prior_type",
        type=str,
        default="not_specified",
        help="Type of prior (default: not_specified)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs for each task (default: 100)",
    )

    parser.add_argument(
        "--start_var_opt",
        type=int,
        default=0,
        help="Epoch at which to start optimizing variance parameters (default: 0)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size to use for training (default: 100)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )

    parser.add_argument(
        "--learning_rate_var",
        type=float,
        default=1e-3,
        help="Learning rate for logvar paramters (default: 1e-3)",
    )

    parser.add_argument(
        "--dropout_rate", type=float, default=0.0, help="Dropout rate (default: 0.0)"
    )

    parser.add_argument(
        "--regularization",
        type=float,
        default=0,
        help="Regularization parameter (default: 0)",
    )

    parser.add_argument(
        "--inducing_points",
        type=int,
        default=0,
        help="Number of BNN inducing points (default: 0)",
    )

    parser.add_argument(
        "--n_marginals",
        type=int,
        default=1,
        help="Number of marginal dimensions to evaluate the KL supremum over (default: 1)",
    )

    parser.add_argument(
        "--n_condition",
        type=int,
        default=0,
        help="Number of conditioning points for modified batch normalization (default: 0)",
    )

    parser.add_argument(
        "--inducing_input_type",
        type=str,
        default="not_specified",
        help="Inducing input selection method (default: not_specified)",
    )

    parser.add_argument(
        "--inducing_input_ood_data",
        nargs="+",
        default=["not_specified"],
        help="Inducing input ood data distribution (default: [not_specified])",
    )

    parser.add_argument(
        "--inducing_input_ood_data_size",
        type=int,
        default=50000,
        help="Size of inducing input ood dataset (default: 50000)",
    )

    parser.add_argument(
        "--kl_scale", type=str, default="1", help="KL scaling factor (default: 1)"
    )

    parser.add_argument(
        "--feature_map_jacobian", action="store_true", default=False, help="Use Jacobian feature map (default: False)"
    )

    parser.add_argument(
        "--feature_map_jacobian_train_only", action="store_true", default=False, help="Do not use Jacobian feature map at evaluation time (default: False)"
    )

    parser.add_argument(
        "--feature_map_type", type=str, default="not_specified", help="Feature map update type (default: not_specified)"
    )

    parser.add_argument(
        "--td_prior_scale", type=float, default=0.0, help="FS-MAP prior penalty scale (default: 0.0)"
    )

    parser.add_argument(
        "--feature_update",
        type=int,
        default=1,
        help="Frequency of feature map updates (default: 1)",
    )

    parser.add_argument(
        "--full_cov", action="store_true", default=False, help="Use full covariance"
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of exp log lik samples (default: 1)",
    )

    parser.add_argument(
        "--n_samples_eval",
        type=int,
        default=0,
        help="Number of evaluation samples (default: 1)",
    )

    parser.add_argument(
        "--tau", type=float, default=1, help="Likelihood precision (default: 1)"
    )

    parser.add_argument(
        "--noise_std", type=float, default=1, help="Likelihood variance (default: 1)"
    )

    parser.add_argument(
        '--ind_lim', type=str, default='ind_-1_1', help='Inducing point range (default: ind_-1_1)'
    )

    parser.add_argument(
        "--logging_frequency",
        type=int,
        default=10,
        help="Logging frequency in number of epochs (default: 10)",
    )

    parser.add_argument(
        "--figsize",
        nargs="+",
        default=[10, 4],
        help="Size of figures (default: (10, 4))",
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    parser.add_argument(
        "--save_path",
        type=str,
        default="debug",
        help="Path to save results (default: debug)",
    )

    parser.add_argument(
        "--save", action="store_true", default=False, help="Save output to file"
    )

    parser.add_argument(
        '--name', type=str, default='', help='Name (default: '')'
    )

    parser.add_argument(
        "--evaluate", action="store_true", default=False, help="Evaluate trained model (default: False)"
    )

    parser.add_argument(
        "--resume_training", action="store_true", default=False, help="Resume training"
    )

    parser.add_argument(
        "--no_final_layer_bias", action="store_true", default=False, help="No bias term in final layer (default: False)"
    )

    parser.add_argument(
        "--extra_linear_layer", action="store_true", default=False, help="additional linear penultimate layer"
    )

    parser.add_argument(
        "--map_initialization", action="store_true", default=False, help="MAP initialization"
    )

    parser.add_argument(
        "--stochastic_linearization", action="store_true", default=False, help="Stochastic linearization"
    )

    parser.add_argument(
        "--grad_flow_jacobian", action="store_true", default=False, help="Gradient flow through Jacobian evaluation (default: False)"
    )

    parser.add_argument(
        '--stochastic_prior_mean', type=str, default='not_specified', help='Stochastic prior mean (default: not_specified)'
    )

    parser.add_argument(
        "--batch_normalization", action="store_true", default=False, help="Batch normalization"
    )

    parser.add_argument(
        '--batch_normalization_mod', type=str, default='not_specified', help='Type of batch normalization (default: not_specified)'
    )

    parser.add_argument(
        "--final_layer_variational", action="store_true", default=False, help="Linear model"
    )

    parser.add_argument(
        '--kl_sup', type=str, default='not_specified', help='Type of KL supremum estimation (default: not_specified)'
    )

    parser.add_argument(
        "--kl_sampled", action="store_true", default=False, help="Use Monte Carlo estimate of KL"
    )

    parser.add_argument(
        "--fixed_inner_layers_variational_var", action="store_true", default=False, help="Fixed inner layer variational variance"
    )

    parser.add_argument(
        "--init_logvar",
        nargs="+",
        default=[0.0, 0.0],
        type=float,
        help="logvar initialization range (default: [0.0,0.0])",
    )

    parser.add_argument(
        "--init_logvar_lin",
        nargs="+",
        default=[0.0, 0.0],
        type=float,
        help="logvar linear layer initialization range (default: [0.0,0.0])",
    )

    parser.add_argument(
        "--init_logvar_conv",
        nargs="+",
        default=[0.0, 0.0],
        type=float,
        help="logvar convolutional layer initialization range (default: [0.0,0.0])",
    )

    parser.add_argument(
        "--perturbation_param", type=float, default=0.01, help="Linearization parameter pertubation parameter (default: 0.01)"
    )

    parser.add_argument(
        "--debug", action="store_true", default=False, help="Debug model"
    )

    parser.add_argument(
        "--logroot", type=str, help="The root result folder that store runs for this type of experiment"
    )

    parser.add_argument(
        "--subdir", type=str, help="The subdirectory in logroot/runs/ corresponding to this run"
    )

    parser.add_argument(
        "--wandb_project", type=str, default="not_specified", help="wanbd project (default: not_specified)"
    )


def define_parser():
    parser = argparse.ArgumentParser()
    add_base_args(parser)
    return parser


def parse_args():
    return define_parser().parse_args()


def process_args(args: argparse.Namespace) -> Dict:
    """
    This is the only place where it is allowed to modify kwargs

    This function should not have side-effect.

    @param args: input arguments
    @return:
    """
    kwargs = vars(args)
    if args.data_ood == "not_specified":
        task = args.data_training
    else:
        task = f"{args.data_training}_{args.data_ood}"
    kwargs["task"] = task

    ind_lim = kwargs['ind_lim'].split('_')[1:]
    ind_lim[0] = float(ind_lim[0])
    ind_lim[1] = float(ind_lim[1])
    kwargs['ind_lim'] = ind_lim

    save_path = args.save_path.rstrip()
    if args.save:
        save_path = (
            f"results/{save_path}/{task}/model_{args.model}__architecture_{args.architecture}__priormean_{args.prior_mean}__"
            f"priorcov_{args.prior_cov}__optimizer_{args.optimizer}__lr_{args.learning_rate}__bs_{args.batch_size}__"
            f"indpoints_{args.inducing_points}__indtype_{args.inducing_input_type}__klscale_{args.kl_scale}__nsamples_{args.n_samples}__"
            # f"SL_{args.stochastic_linearization}__MI_{args.map_initialization}__"
            f"nmarginal_{args.n_marginals}"
            f"tau_{args.tau}__indlim_{ind_lim[0]}_{ind_lim[1]}__reg_{args.regularization}__seed_{args.seed}__"
        )
        i = 1
        while os.path.exists(f"{save_path}{i}") or os.path.exists(
            f"{save_path}{i}__complete"
        ):
            i += 1
        save_path = f"{save_path}{i}"
    kwargs["save_path"] = save_path

    if "mlp" in kwargs["model"]:
        # This assumes the string starts with 'fc_' followed by the number of hidden units for each layer
        kwargs["architecture_arg"] = kwargs["architecture"]
        kwargs["architecture"] = list(map(int, kwargs["architecture"].split("_")[1:]))

    if kwargs["n_condition"] == 0:
        kwargs["n_condition"] = kwargs["batch_size"]
    if kwargs["td_prior_scale"] == 0.0:
        kwargs["td_prior_scale"] = float(kwargs["inducing_points"])
    if kwargs["prior_covs"][0] != 0.0:
        prior_cov = []
        for i in range(len(kwargs["prior_covs"])):
            prior_cov.append(kwargs["prior_covs"][i])
        kwargs["prior_cov"] = prior_cov

    if kwargs["feature_map_type"] == "learned_nograd":
        kwargs["grad_flow_jacobian"] = False
    elif kwargs["feature_map_type"] == "learned_grad":
        kwargs["grad_flow_jacobian"] = True

    kwargs["init_logvar_minval"] = float(kwargs["init_logvar"][0])
    kwargs["init_logvar_maxval"] = float(kwargs["init_logvar"][1])
    kwargs["init_logvar_lin_minval"] = float(kwargs["init_logvar_lin"][0])
    kwargs["init_logvar_lin_maxval"] = float(kwargs["init_logvar_lin"][1])
    kwargs["init_logvar_conv_minval"] = float(kwargs["init_logvar_conv"][0])
    kwargs["init_logvar_conv_maxval"] = float(kwargs["init_logvar_conv"][1])
    kwargs["figsize"] = tuple(kwargs["figsize"])
    kwargs["model_type"] = kwargs.pop("model")
    kwargs["n_inducing_inputs"] = kwargs.pop("inducing_points")
    kwargs["inducing_inputs_bound"] = kwargs.pop("ind_lim")

    return kwargs


def run(args):
    kwargs = process_args(args)
    # all subsequent code should not modify kwargs

    # alternative logging
    if kwargs["logroot"]:
        kwargs["run_folder"] = str(create_logdir(kwargs["logroot"], kwargs["subdir"]))

    if kwargs["save"]:
        # Automatically makes parent directories
        os.makedirs(f"{kwargs['save_path']}/figures", exist_ok=True)
        config_header = kwargs.keys()
        with open(f"{kwargs['save_path']}/config.csv", "a") as config_file:
            config_writer = csv.DictWriter(config_file, fieldnames=config_header)
            config_writer.writeheader()
            config_writer.writerow(kwargs)
            config_file.close()

    if kwargs["save"] and not kwargs["debug"]:
        orig_stdout = sys.stdout
        stdout_file = open(f"{kwargs['save_path']}/stdout.txt", "w")
        sys.stdout = stdout_file

    if kwargs["debug"]:
        print(f"\nDevice: {xla_bridge.get_backend().platform}\n")

    print(
        "Input arguments:\n", json.dumps(kwargs, indent=4, separators=(",", ":")), "\n"
    )

    kwargs['kwargs'] = copy(kwargs)

    main(kwargs)

    if kwargs["save"] and not kwargs["debug"]:
        sys.stdout = orig_stdout
        stdout_file.close()


if __name__ == "__main__":
    run(parse_args())

import argparse
tf_cpu_only = True  # TODO: check how this affects determinism -- keep set to False
if tf_cpu_only:
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], "GPU")
    print('WARNING: TensorFlow is set to only use CPU.')
from jax.lib import xla_bridge
print("Jax is running on", xla_bridge.get_backend().platform)

from fsvi_cl.src_cl.trainers.args_cl import add_cl_args
from fsvi_cl.run import run as cl_run
from run_base import run as base_run
from run_base import add_base_args


def define_parser():
    parser = argparse.ArgumentParser(description="Function-Space Variation Inference")
    subparsers = parser.add_subparsers(required=True, dest="command")
    add_cl_args(subparsers.add_parser("cl"))
    add_base_args(subparsers.add_parser("base"))
    return parser


def parse_args():
    return define_parser().parse_args()


def cli():
    args = parse_args()

    if args.command == "cl":
        cl_run(args)
    elif args.command == "base":
        base_run(args)
    else:
        raise NotImplementedError(f"Unknown command {args.command}")


if __name__ == "__main__":
    cli()

# FSVI
Function-space Variational Inference in Bayesian Neural Networks

## Installation

```
$ conda env update -f environment.yml
$ conda activate fsvi
```

This environment includes all the dependencies. However, make sure you also install CUDA and cuDNN. Unlike Torch, Jax doesn't provide fat binaries with linked libraries and it searches for a system-wide CUDA installation.

In order to be able to import modules on Slurm, use `pip install -e .` to get access to `fsvi` executable.

## Command Line Executable

After installing the `fsvi` executable by running `pip install -e .` in the project root directory, you can call the runners by using `fsvi` instead of using python which requires specifying the path to the runner file.

To invoke the runner for the base repo, 

```bash
fsvi base <options>
```

## Reproduce Experiments

View Jupter notebooks in directory

```bash
reproduce_experiments
```
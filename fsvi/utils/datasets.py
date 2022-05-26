import getpass
import os
import urllib.request
from os import path
from pathlib import Path
from typing import NamedTuple, Iterator

import jax.numpy as jnp
import numpy as np
import seqtools
import sklearn
import sklearn.datasets
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets as torch_datasets
from torchvision import transforms

from scipy.io import loadmat
import pyreadr

from sklearn import preprocessing
from sklearn import model_selection

from PIL import Image

try:
    import uncertainty_baselines as ub
except:
    print("WARNING: uncertainty_baselines could not be loaded.")

try:
    from bayesian_benchmarks.data import get_regression_data
    from bayesian_benchmarks.database_utils import Database
except:
    print("WARNING: bayesian_benchmarks could not be loaded.")

if getpass.getuser() == "ANON":
    download = False
    download_path = "/scratch/data"
else:
    download = True
    download_path = "not_specified"

download = True

datasets_with_dataloader = ["mnist", "notmnist", "fashionmnist", "cifar10", "cifar10_noaugmentation", "svhn", "ihdp"]
classification_tasks = ["mnist", "notmnist", "fashionmnist", "cifar10", "cifar10_noaugmentation", "svhn"]
regression_tasks = ["ihdp"]

_DATA = "/tmp/jax_example_data/"

dtype_default = jnp.float32
torch.manual_seed(0)


def _one_hot(x, k, dtype=dtype_default):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def collate_fn(batch):
    inputs = np.stack([x for x, _ in batch])
    targets = np.stack([y for _, y in batch])
    return inputs, targets


def load_data(
    model_type,
    data_training,
    data_ood,
    inducing_input_ood_data,
    inducing_input_ood_data_size,
    batch_size,
    seed,
    val_frac=0.0,  # TODO: implement validation set
    return_testloader=False,
    data_inducing=None,  # TODO: data incuding is actually not used
    **kwargs,
):
    # SELECT TRAINING AND TEST DATA LOADERS
    if (
        data_training in datasets_with_dataloader
    ):  # TODO: implement methods that return dataloaders for all datsets
        if data_training == "mnist":
            input_dim, output_dim, train_dataset, test_dataset = get_MNIST()
        if data_training == "fashionmnist":
            input_dim, output_dim, train_dataset, test_dataset = get_FashionMNIST()
        if data_training == "cifar10":
            input_dim, output_dim, train_dataset, test_dataset = get_CIFAR10(augmentation_train=True, augmentation_test=False)
        if data_training == "cifar10_noaugmentation":
            input_dim, output_dim, train_dataset, test_dataset = get_CIFAR10(augmentation_train=False, augmentation_test=False)
        if data_training == "svhn":
            input_dim, output_dim, train_dataset, test_dataset = get_SVHN()
        if data_training == 'ihdp':
            input_dim, output_dim, train_dataset, test_dataset = get_ihdp(seed)

        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=len(test_dataset), shuffle=False
        )

        if data_training in classification_tasks:
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                if i == 0:
                    x_train = np.array(inputs).transpose([0, 2, 3, 1])
                    y_train = _one_hot(np.array(labels), 10)
                else:
                    x_train = np.concatenate(
                        [x_train, np.array(inputs).transpose([0, 2, 3, 1])], 0
                    )
                    y_train = np.concatenate([y_train, _one_hot(np.array(labels), 10)], 0)

            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                x_test = np.array(inputs).transpose([0, 2, 3, 1])
                y_test = _one_hot(np.array(labels), 10)
        else:
            for i, data in enumerate(trainloader, 0):
                inputs, targets = data
                if i == 0:
                    x_train = np.array(inputs)
                    y_train = np.array(targets)
                else:
                    x_train = np.concatenate(
                        [x_train, np.array(inputs)], 0
                    )
                    y_train = np.concatenate([y_train, targets], 0)

            for i, data in enumerate(testloader, 0):
                inputs, targets = data
                x_test = np.array(inputs)
                y_test = np.array(targets)

    elif data_training == "dr":
        (
            image_dim,
            output_dim,
            trainloader,
            x_train,
            y_train,
            x_test,
            y_test,
        ) = get_diabetic_retinopathy(batch_size=batch_size)

    else:
        print("WARNING: DATALOADER HAS NOT BEEN TESTED")  # TODO: remove once tested
        if "uci" in data_training:
            data_training = data_training.split("uci_", 1)[1].split("_", 1)[0]
            data = get_regression_data(data_training, split=seed)
            x_train, y_train, x_test, y_test = (
                data.x_train,
                data.y_train,
                data.x_test,
                data.y_test,
            )
            y_train = y_train.reshape([-1, 1])
            input_dim = x_train.shape[-1]
            output_dim = 1

        elif "offline_rl" in data_training:
            data = np.load(
                "/scratch-ssd/ANON/deployment/testing/data/large/offpolicy_hand_data/door2_sparse.npy",
                allow_pickle=True,
            )
            if type(data[0]["observations"][0]) is dict:
                # Convert to just the states
                for traj in data:
                    traj["observations"] = [
                        t["state_observation"] for t in traj["observations"]
                    ]
            X = np.array(
                [j for i in [traj["observations"] for traj in data] for j in i],
                dtype=dtype_default,
            )
            Y = np.array(
                [j for i in [traj["actions"] for traj in data] for j in i],
                dtype=dtype_default,
            )
            input_dim = X.shape[-1]  # TODO: implement
            output_dim = 1

        elif data_training == "snelson":
            x_train, y_train, x_test, inducing_inputs_, noise_std = snelson(
                n_test=1000, x_test_lim=10, standardize_x=True, standardize_y=False
            )
            y_train = y_train.reshape([-1, 1])
            input_dim = x_train.shape[-1]
            output_dim = 1

        elif data_training == "two_moons":
            x_train, y_train = sklearn.datasets.make_moons(
                n_samples=200, shuffle=True, noise=0.2, random_state=seed
            )
            y_train = _one_hot(y_train, 2)
            input_dim = x_train.shape[-1]
            output_dim = y_train.shape[-1]

            h = 0.25
            test_lim = 3
            # x_min, x_max = x_train[:, 0].min() - test_lim, x_train[:, 0].max() + test_lim
            # y_min, y_max = x_train[:, 1].min() - test_lim, x_train[:, 1].max() + test_lim
            x_min, x_max = (
                x_train[:, 0].min() - test_lim,
                x_train[:, 0].max() + test_lim,
            )
            y_min, y_max = (
                x_train[:, 1].min() - test_lim,
                x_train[:, 1].max() + test_lim,
            )
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            x_test = np.vstack((xx.reshape(-1), yy.reshape(-1))).T

            h = 0.25
            test_lim = 7
            x_wide_min, x_wide_max = (
                x_train[:, 0].min() - test_lim,
                x_train[:, 0].max() + test_lim,
            )
            y_wide_min, y_wide_max = (
                x_train[:, 1].min() - test_lim,
                x_train[:, 1].max() + test_lim,
            )
            xx_wide, yy_wide = np.meshgrid(
                np.arange(x_wide_min, x_wide_max, h),
                np.arange(y_wide_min, y_wide_max, h),
            )
            x_test_wide = np.vstack((xx_wide.reshape(-1), yy_wide.reshape(-1))).T

        # DEFINE NUMPY TRAINLOADER  # TODO: test implementation
        n_train = x_train.shape[0]
        train_dataset = seqtools.collate([x_train, y_train])
        if batch_size == 0:
            trainloader_ = seqtools.batch(train_dataset, n_train, collate_fn=collate_fn)
        else:
            trainloader_ = seqtools.batch(
                train_dataset, batch_size, collate_fn=collate_fn
            )

        trainloader = []
        for i, data in enumerate(trainloader_, 0):
            x_batch = np.array(data[0], dtype=dtype_default)
            y_batch = np.array(data[1], dtype=dtype_default)
            trainloader.append([x_batch, y_batch])

    n_train = x_train.shape[0]
    n_batches = n_train // batch_size

    input_shape = list(x_train.shape)
    input_shape[0] = 1

    permutation = np.random.permutation(n_train)
    x_train_permuted = x_train[permutation]
    y_train_permuted = y_train[permutation]

    # SELECT OUT-OF-DISTRIBUTION DATA LOADER
    if data_ood != "not_specified":
        x_ood_list = []
        y_ood_list = []
        x_inducing_input_ood_list = []
        y_inducing_input_ood_list = []

        for dataset in data_ood:
            if dataset == "mnist":
                _, _, _, ood_dataset = get_MNIST()
            elif dataset == "notmnist":
                _, _, _, ood_dataset = get_NotMNIST()
            elif dataset == "kmnist":
                _, _, _, ood_dataset = get_KMNIST()
            elif dataset == "fashionmnist":
                _, _, _, ood_dataset = get_FashionMNIST()
            elif dataset == "cifar10":
                _, _, _, ood_dataset = get_CIFAR10(augmentation_train=False, augmentation_test=False)
            elif dataset == "cifar10_augmentation":
                _, _, _, ood_dataset = get_CIFAR10(augmentation_train=False, augmentation_test=True)
            elif dataset == "cifar100":
                _, _, _, ood_dataset = get_CIFAR100()
            elif dataset == "miniplaces":
                _, _, _, ood_dataset = get_Miniplaces()
            elif dataset == "fake":
                _, _, _, ood_dataset = get_Fake()
            elif dataset == "svhn":
                _, _, _, ood_dataset = get_SVHN()
            elif dataset == "corrupted_cifar10":
                x_ood, y_ood = get_CorruptedCIFAR10()
                y_ood = _one_hot(y_ood, 10)
            elif dataset == "catsvsdogs":
                x_ood, _ = get_CatsVsDogs()
            elif dataset == "train_monochrome":
                x_ood = get_Monochrome(x_train)
                y_ood = jnp.ones((x_ood.shape[0], 10))
            elif dataset == "dr":
                x_ood = x_test  # TODO: implement
            else:
                raise ValueError(f"Out-of-distribution dataset not recognized: {dataset}")

            # TODO: messy, clean up:
            if dataset != "dr" and dataset != "corrupted_cifar10" and dataset != "catsvsdogs" and dataset != "train_monochrome":
                if dataset != "cifar10_augmentation":
                    ood_dataset.transform = test_dataset.transform
                ood_dataset_size = min(len(ood_dataset), 50000)
                oodloader = torch.utils.data.DataLoader(
                    ood_dataset, batch_size=ood_dataset_size, shuffle=True
                )

                for i, data in enumerate(oodloader, 0):
                    inputs, labels = data
                    x_ood = jnp.array(inputs, dtype=dtype_default).transpose([0, 2, 3, 1])
                    y_ood = _one_hot(jnp.array(labels, dtype=dtype_default), 10)
                    break  # needed for datasets with more datapoints than specified in ood_dataset_size

            # import matplotlib.pyplot as plt
            # for i in range(10):
            #     plt.imshow(jnp.array(inputs, dtype=dtype_default).transpose([0, 2, 3, 1])[i, :, :, :])
            #     plt.show()

            if "mlp" in model_type:
                x_ood = x_ood.reshape([x_ood.shape[0], -1])
                x_test = x_test.reshape([x_test.shape[0], -1])
                x_train = x_train.reshape([x_train.shape[0], -1])
                x_train_permuted = x_train_permuted.reshape([x_train_permuted.shape[0], -1])

            permutation = np.random.permutation(x_ood.shape[0])
            x_ood_permuted = x_ood[permutation]
            y_ood_permuted = y_ood[permutation]

            if dataset in inducing_input_ood_data:
                x_inducing_input_ood_list.append(x_ood_permuted)
                y_inducing_input_ood_list.append(y_ood_permuted)

            x_ood_list.append(x_ood_permuted)
            y_ood_list.append(y_ood_permuted)
    else:
        x_ood_list = []
        y_ood_list = []
        x_inducing_input_ood_list = []
        y_inducing_input_ood_list = []

    min_size_inducing = inducing_input_ood_data_size
    for data in x_inducing_input_ood_list:
        n_points = data.shape[0]
        size_inducing = min(min_size_inducing, n_points)

    # Ensure all inducing sets are the same size to avoid unbalanced samples
    for i in range(len(x_inducing_input_ood_list)):
        x_inducing_input_ood_list[i] = x_inducing_input_ood_list[i][:size_inducing]

    # for i in range(10):
    #     plt.imshow(jnp.array(x_train_permuted, dtype=dtype_default)[i, :, :, :])
    #     plt.show()

    if (
        data_inducing is not None
    ):  # TODO: implement methods that return dataloaders for all datsets
        if data_inducing == "mnist":
            _, _, inducing_dataset, _ = get_MNIST()
        if data_inducing == "notmnist":
            _, _, inducing_dataset, _ = get_NotMNIST()
        if data_inducing == "emnist":
            _, _, inducing_dataset, _ = get_EMNIST()
        if data_inducing == "kmnist":
            _, _, inducing_dataset, _ = get_KMNIST()
        if data_inducing == "fashionmnist":
            _, _, inducing_dataset, _ = get_FashionMNIST()
        if data_inducing == "cifar10":
            _, _, inducing_dataset, _ = get_CIFAR10()
        if data_inducing == "cifar100":
            _, _, inducing_dataset, _ = get_CIFAR100()
        if data_inducing == "svhn":
            _, _, inducing_dataset, _ = get_SVHN()

        inducing_dataset.transform = train_dataset.transform
        inducingloader = torch.utils.data.DataLoader(
            inducing_dataset, batch_size=len(inducing_dataset), shuffle=False
        )

        for i, data in enumerate(inducingloader, 0):
            inputs, labels = data
            x_inducing = np.array(inputs).transpose([0, 2, 3, 1])

    if return_testloader == False and data_inducing is None:
        return (
            trainloader,
            x_train_permuted,
            y_train_permuted,
            x_test,
            y_test,
            x_ood_list,
            y_ood_list,
            x_inducing_input_ood_list,
            y_inducing_input_ood_list,
            input_shape,
            input_dim,
            output_dim,
            n_train,
            n_batches,
        )
    elif return_testloader == False and data_inducing is not None:
        return (
            trainloader,
            x_train_permuted,
            y_train_permuted,
            x_test,
            y_test,
            x_ood_list,
            y_ood_list,
            x_inducing_input_ood_list,
            y_inducing_input_ood_list,
            x_inducing,
            input_shape,
            input_dim,
            output_dim,
            n_train,
            n_batches,
        )
    else:
        return (
            trainloader,
            testloader,
            x_train_permuted,
            y_train_permuted,
            x_test,
            y_test,
            x_ood_list,
            y_ood_list,
            x_inducing_input_ood_list,
            y_inducing_input_ood_list,
            input_shape,
            input_dim,
            output_dim,
            n_train,
            n_batches,
        )


def _download(url, filename):
    """Download a url to a file in the JAX data temp directory."""
    if not path.exists(_DATA):
        os.makedirs(_DATA)
    out_file = path.join(_DATA, filename)
    if not path.isfile(out_file):
        urllib.request.urlretrieve(url, out_file)
        print("downloaded {} to {}".format(url, _DATA))


def _partial_flatten(x):
    """Flatten all but the first dimension of an ndarray."""
    return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=dtype_default):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def get_Monochrome(inputs):
    input_shape = list(inputs.shape)
    input_shape[0] = 1
    n_samples = min(inputs.shape[0], 10000)

    permutation = np.random.permutation(inputs.shape[0])
    inputs_permuted = inputs[permutation, :]

    if len(input_shape) == 4 and input_shape[-1] == 1:
        # Select random pixel values
        random_pixels = np.random.choice(
            a=inputs_permuted.flatten(),
            size=(n_samples,),
            replace=False,
        )
        pixel_samples = jnp.array(jnp.transpose(
            random_pixels * jnp.ones(input_shape), (3, 1, 2, 0)), dtype=dtype_default
        )
    elif len(input_shape) == 4 and input_shape[-1] > 1:
        image_dim = input_shape[1]
        num_channels = input_shape[-1]
        pixel_samples_list = []
        for channel in range(num_channels):
            # Select random pixel values for given channel
            random_pixels = np.random.choice(
                a=inputs_permuted[:, :, :, channel].flatten(),
                size=(n_samples,),
                replace=False,
            )
            _pixel_samples = jnp.array(jnp.transpose(
                random_pixels * jnp.ones([1, image_dim, image_dim, 1], dtype=dtype_default), (3, 1, 2, 0)), dtype=dtype_default
            )
            pixel_samples_list.append(_pixel_samples)
        pixel_samples = jnp.concatenate(pixel_samples_list, axis=3)
    else:
        # Select random pixel values
        random_pixels = np.random.choice(
            a=inputs_permuted.flatten(),
            size=input_shape[0],
            replace=False,
        )[:, None]
        pixel_samples = jnp.array(random_pixels * jnp.ones(input_shape[-1]), dtype=dtype_default)

    return pixel_samples


def get_SVHN(root="./data/SVHN"):
    image_dim = 32
    num_classes = 10

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(image_dim, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = torch_datasets.SVHN(
        root, split="train", transform=train_transform, download=download
    )
    test_dataset = torch_datasets.SVHN(
        root, split="test", transform=test_transform, download=download
    )

    return image_dim, num_classes, train_dataset, test_dataset


def get_CorruptedCIFAR10(root="./data/"):
    ds = tfds.load(
        "cifar10_corrupted", data_dir=root + "data/cifar10_corrupted", split="test"
    )
    ds = tfds.as_numpy(ds)
    images = []
    labels = []
    for ex in ds:
        images.append(ex["image"])
        labels.append(ex["label"])

    images = np.stack(images, axis=0)
    labels = np.stack(labels, axis=0)

    data_transform = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255),
            tf.keras.layers.experimental.preprocessing.Normalization(
                mean=(0.4914, 0.4822, 0.4465), variance=(0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    images = data_transform(images)

    return jnp.array(images, dtype=dtype_default), labels


def get_CatsVsDogs(root="./data/"):
    builder = tfds.builder("cats_vs_dogs", data_dir=root + "data/cats_vs_dogs")
    builder.download_and_prepare()
    ds = builder.as_dataset(split='train', as_supervised=True)

    images = []
    labels = []

    for image, label in tfds.as_numpy(ds):
        images.append(tf.image.resize(image, size=[32,32]))
        labels.append(label)

    # import matplotlib.pyplot as plt
    # for i in range(9):
    #     plt.subplot(330 + 1 + i)
    #     plt.imshow(images[i] / 255)

    images = np.stack(images, axis=0)
    labels = np.stack(labels, axis=0)

    data_transform = tf.keras.Sequential(
        [
            # tf.keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255.),
            tf.keras.layers.experimental.preprocessing.Normalization(
                mean=(0.4914, 0.4822, 0.4465), variance=(0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    images = data_transform(images)

    return jnp.array(images, dtype=dtype_default), labels


def get_CIFAR10(augmentation_train=True, augmentation_test=False, root="./data/CIFAR10"):
    image_dim = 32
    num_classes = 10

    if augmentation_train:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(image_dim, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    if augmentation_test:
        test_transform = transforms.Compose(
            [
                transforms.RandomCrop(image_dim, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(32),  # won't change CIFAR-10 data, but may be needed for OOD data
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    else:
        test_transform = transforms.Compose(
            [
                transforms.Resize(32),  # won't change CIFAR-10 data, but may be needed for OOD data
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    train_dataset = torch_datasets.CIFAR10(
        root, train=True, transform=train_transform, download=download
    )
    test_dataset = torch_datasets.CIFAR10(
        root, train=False, transform=test_transform, download=download
    )

    return image_dim, num_classes, train_dataset, test_dataset


def get_CIFAR100(root="./data/CIFAR100"):
    image_dim = 32
    num_classes = 10

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(image_dim, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_dataset = torch_datasets.CIFAR100(
        root, train=True, transform=train_transform, download=download
    )
    test_dataset = torch_datasets.CIFAR100(
        root, train=False, transform=test_transform, download=download
    )

    return image_dim, num_classes, train_dataset, test_dataset


def get_Miniplaces(root=download_path):
    image_dim = 32
    num_classes = 10

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            # transforms.Scale(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ],

    )

    download = not Path(root+"/miniplaces").is_dir()
    if download:
        torchvision.datasets.utils.download_and_extract_archive(
            'https://dissect.csail.mit.edu/datasets/miniplaces.zip',
            download_path,
            md5='bfabeb497c7eca01c74cd8441a9ac108')

    train_dataset = torch_datasets.ImageFolder(
        root+"/miniplaces"+"/train", transform=transform,
    )
    test_dataset = train_dataset

    return image_dim, num_classes, train_dataset, test_dataset


def get_Fake():
    image_dim = 32
    num_classes = 10

    transform = transforms.Compose(
        [
            transforms.RandomCrop(image_dim, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset = torch_datasets.FakeData(
        size=60000, image_size=(3,32,32), transform=transform,
    )

    return image_dim, num_classes, [], dataset


def get_MNIST(root="./data/"):
    image_dim = 28
    num_classes = 10

    transform_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]

    transform = transforms.Compose(transform_list)

    train_dataset = torch_datasets.MNIST(
        root, train=True, download=download, transform=transform
    )
    test_dataset = torch_datasets.MNIST(
        root, train=False, download=download, transform=transform
    )

    return image_dim, num_classes, train_dataset, test_dataset


def get_EMNIST(root="./data/"):
    image_dim = 28
    num_classes = 10

    transform_list = [
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ]

    transform = transforms.Compose(transform_list)

    train_dataset = torch_datasets.EMNIST(
        root, train=True, download=download, transform=transform
    )
    test_dataset = torch_datasets.EMNIST(
        root, train=False, download=download, transform=transform
    )

    return image_dim, num_classes, train_dataset, test_dataset


def get_KMNIST(root="./data/"):
    image_dim = 28
    num_classes = 10

    transform_list = [
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ]

    transform = transforms.Compose(transform_list)

    train_dataset = torch_datasets.KMNIST(
        root, train=True, download=download, transform=transform
    )
    test_dataset = torch_datasets.KMNIST(
        root, train=False, download=download, transform=transform
    )

    return image_dim, num_classes, train_dataset, test_dataset


def get_FashionMNIST(root="./data/"):
    image_dim = 28
    num_classes = 10

    train_transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(30),
            # transforms.RandomCrop(image_dim, padding=4),
            # transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.2861,), (0.3530,)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2861,), (0.3530,)),
        ]
    )

    train_dataset = torch_datasets.FashionMNIST(
        root, train=True, download=download, transform=train_transform
    )
    test_dataset = torch_datasets.FashionMNIST(
        root, train=False, download=download, transform=test_transform
    )

    return image_dim, num_classes, train_dataset, test_dataset


class NotMNIST(Dataset):
    # Download from https://web.archive.org/web/20181212081522/http://yaroslavvb.com/upload/notMNIST/notMNIST_small.mat
    def __init__(self, root, transform=None):
        root = os.path.expanduser(root)
        self.transform = transform
        data_dict = loadmat(os.path.join(root, "notMNIST_small.mat"))
        self.data = torch.tensor(
            data_dict["images"].transpose(2, 0, 1), dtype=torch.uint8
        ).unsqueeze(1)
        self.targets = torch.tensor(data_dict["labels"], dtype=torch.int64)
    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            img = Image.fromarray(img.squeeze().numpy(), mode="L")
            img = self.transform(img)
        return img, target
    def __len__(self):
        return len(self.data)


def get_NotMNIST(root="./data/NotMNIST"):
    image_dim = 28
    num_classes = 10
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4254,), (0.4586,))]
    )
    test_dataset = NotMNIST(root, transform=transform)
    return image_dim, num_classes, None, test_dataset


def get_ihdp(seed, root="./data/"):
    input_dim = 26
    output_dim = 1

    train_dataset = IHDP(root=root, split="train", mode="mu", seed=seed, hidden_confounding=False)
    val_dataset = IHDP(root=root, split="valid", mode="mu", seed=seed, hidden_confounding=False)
    test_dataset = IHDP(root=root, split="test", mode="mu", seed=seed, hidden_confounding=False)

    return input_dim, output_dim, train_dataset, val_dataset


def get_diabetic_retinopathy(
    batch_size, data_dir="/scratch/data/diabetic-retinopathy-detection"
):
    image_dim = 128
    input_shape = [1, image_dim, image_dim, 3]
    output_dim = 2
    loader_batch_size = 5000

    eval_batch_size = 10000
    use_validation = False

    dataset_train_builder = ub.datasets.get(
        "diabetic_retinopathy_detection", split="train", data_dir=data_dir
    )
    trainloader = dataset_train_builder.load(batch_size=loader_batch_size)

    dataset_validation_builder = ub.datasets.get(
        "diabetic_retinopathy_detection",
        split="validation",
        data_dir=data_dir,
        is_training=not use_validation,
    )
    validation_batch_size = eval_batch_size if use_validation else batch_size
    valloader = dataset_validation_builder.load(batch_size=validation_batch_size)
    trainloader = trainloader.concatenate(valloader)

    dataset_test_builder = ub.datasets.get(
        "diabetic_retinopathy_detection", split="test", data_dir=data_dir
    )
    testloader = dataset_test_builder.load(batch_size=eval_batch_size)

    print("Loading diabetic retinopathy dataset. This may take a few minutes.")
    train_iterator = iter(trainloader)
    ds_info = tfds.builder("diabetic_retinopathy_detection").info
    loader_n_batches = ds_info.splits["train"].num_examples // loader_batch_size
    for i in range(loader_n_batches):
        data = next(train_iterator)
        if i == 0:
            x_train = data["features"]._numpy().transpose(0, 3, 1, 2)
            y_train = data["labels"]._numpy()
        else:
            x_train = np.concatenate(
                [x_train, data["features"]._numpy().transpose(0, 3, 1, 2)], 0
            )
            y_train = np.concatenate([y_train, data["labels"]._numpy()], 0)

    x_train = x_train.transpose(0, 2, 3, 1)
    y_train = _one_hot(y_train, output_dim)

    train_dataset = seqtools.collate([x_train, y_train])
    trainloader = seqtools.batch(train_dataset, batch_size, collate_fn=collate_fn)

    test_iterator = iter(testloader)
    data = next(test_iterator)
    x_test = data["features"]._numpy()
    y_test = _one_hot(data["labels"]._numpy(), output_dim)

    return (
        image_dim,
        output_dim,
        trainloader,
        x_train,
        y_train,
        x_test,
        y_test,
    )


def snelson(n_test=500, x_test_lim=2.5, standardize_x=False, standardize_y=False):
    def _load_toydata(filename):
        try:
            with open(f"data/snelson/{filename}", "r") as f:
                return np.array(
                    [dtype_default(i) for i in f.read().strip().split("\n")]
                )
        except Exception as e:
            print(
                f"Error: {e.args[0]}\n\nWorking directory needs to be set to repository root."
            )

    x_train = _load_toydata("train_inputs")
    y_train = _load_toydata("train_outputs")

    mask = ((x_train < 1.5) | (x_train > 3)).flatten()
    x_train = x_train[mask]
    y_train = y_train[mask]

    idx = np.argsort(x_train)
    x_train = x_train[idx]
    y_train = y_train[idx]

    if standardize_x:
        x_train = (x_train - x_train.mean(0)) / x_train.std(0)
    if standardize_y:
        y_train = (y_train - y_train.mean(0)) / y_train.std(0)

    x_test = np.linspace(-x_test_lim, x_test_lim, n_test)[:, None]
    x1, x2, x3, x4 = x_train.min(), -0.9503225, -0.11031368, x_train.max()
    inducing_inputs_ = np.concatenate(
        [
            x_test[(x_test < x1)],
            x_test[np.logical_and(x_test > x2, x_test < x3)],
            x_test[(x_test > x4)],
        ]
    )[:, None]

    noise_std = 0.286

    return x_train[:, None], y_train[:, None], x_test, inducing_inputs_, noise_std


def load_solar(n_test=500, standardize_x=False, standardize_y=False):
    data = np.genfromtxt("data/solar.txt", delimiter=",")

    x = data[:, 0:1]
    y = data[:, 2:3]

    # remove some chunks of data
    x_test, y_test = [], []

    intervals = ((1620, 1650), (1700, 1720), (1780, 1800), (1850, 1870), (1930, 1950))

    for low, up in intervals:
        ind = np.logical_and(X.flatten() > low, x.flatten() < up)
        x_test.append(x[ind])
        y_test.append(y[ind])
        x = np.delete(x, np.where(ind)[0], axis=0)
        y = np.delete(y, np.where(ind)[0], axis=0)

    if standardize_x:
        x_train = (x - x.mean(0)) / x.std(0)
    if standardize_y:
        y_train = (y - y.mean(0)) / y.std(0)

    # x_test, y_test = np.vstack(x_test), np.vstack(y_test)
    x_test = np.linspace(-1, 2, n_test)[:, None]

    noise_std = 0.01

    return x_train, y_train, x_test, noise_std


def in_between_uncertainty(
    rng_key, n_train=50, n_test=200, x_test_lim=2, loc=1, noise_std=0.1
):
    c1 = tfd.Normal(loc=[-loc, -loc], scale=[noise_std, noise_std])
    c2 = tfd.Normal(loc=[loc, loc], scale=[noise_std, noise_std])

    s1 = c1.sample(n_train, seed=rng_key)
    s2 = c2.sample(n_train, seed=rng_key)

    x_train = np.concatenate([s1[:, 0:1], s2[:, 0:1]], axis=0)
    y_train = np.concatenate([s1[:, 1:2], s2[:, 1:2]], axis=0)

    x_test = np.linspace(-x_test_lim, x_test_lim, n_test)[:, None]

    x1 = s1[:, 0:1].min()
    x2 = s1[:, 0:1].max()
    x3 = s2[:, 0:1].min()
    x4 = s2[:, 0:1].max()

    inducing_inputs_ = np.concatenate(
        [
            x_test[(x_test < x1)],
            x_test[np.logical_and(x_test > x2, x_test < x3)],
            x_test[(x_test > x4)],
        ]
    )[:, None]

    # inducing_inputs_ = np.linspace(-0.4, 0.4, n_test)[:, None]

    return x_train, y_train, x_test, inducing_inputs_, noise_std


class IHDP(Dataset):
    def __init__(self, root, split, mode, seed, hidden_confounding, beta_u=None):

        _CONTINUOUS_COVARIATES = [
            "bw",
            "b.head",
            "preterm",
            "birth.o",
            "nnhealth",
            "momage",
        ]

        _BINARY_COVARIATES = [
            "sex",
            "twin",
            "mom.lths",
            "mom.hs",
            "mom.scoll",
            "cig",
            "first",
            "booze",
            "drugs",
            "work.dur",
            "prenatal",
            "ark",
            "ein",
            "har",
            "mia",
            "pen",
            "tex",
            "was",
        ]

        _HIDDEN_COVARIATE = [
            "b.marr",
        ]

        root = Path(root)
        df = pyreadr.read_r(str(root / "ihdp.RData"))["ihdp"]
        # Make observational as per Hill 2011
        df = df[~((df["treat"] == 1) & (df["momwhite"] == 0))]
        df = df[
            _CONTINUOUS_COVARIATES + _BINARY_COVARIATES + _HIDDEN_COVARIATE + ["treat"]
        ]
        # Standardize continuous covariates
        df[_CONTINUOUS_COVARIATES] = preprocessing.StandardScaler().fit_transform(
            df[_CONTINUOUS_COVARIATES]
        )
        # Generate response surfaces
        rng = np.random.default_rng(seed)
        x = df[_CONTINUOUS_COVARIATES + _BINARY_COVARIATES]
        u = df[_HIDDEN_COVARIATE]
        t = df["treat"]
        beta_x = rng.choice(
            [0.0, 0.1, 0.2, 0.3, 0.4], size=(24,), p=[0.6, 0.1, 0.1, 0.1, 0.1]
        )
        beta_u = (
            rng.choice(
                [0.1, 0.2, 0.3, 0.4, 0.5], size=(1,), p=[0.2, 0.2, 0.2, 0.2, 0.2]
            )
            if beta_u is None
            else np.asarray([beta_u])
        )
        mu0 = np.exp((x + 0.5).dot(beta_x) + (u + 0.5).dot(beta_u))
        df["mu0"] = mu0
        mu1 = (x + 0.5).dot(beta_x) + (u + 0.5).dot(beta_u)
        omega = (mu1[t == 1] - mu0[t == 1]).mean(0) - 4
        mu1 -= omega
        df["mu1"] = mu1
        eps = rng.normal(size=t.shape)
        y0 = mu0 + eps
        df["y0"] = y0
        y1 = mu1 + eps
        df["y1"] = y1
        y = t * y1 + (1 - t) * y0
        df["y"] = y
        # Train test split
        df_train, df_test = model_selection.train_test_split(
            df, test_size=0.1, random_state=seed
        )
        self.mode = mode
        self.split = split
        # Set x, y, and t values
        self.y_mean = (
            df_train["y"].to_numpy(dtype="float32").mean(keepdims=True)
            if mode == "mu"
            else np.asarray([0.0], dtype="float32")
        )
        self.y_std = (
            df_train["y"].to_numpy(dtype="float32").std(keepdims=True)
            if mode == "mu"
            else np.asarray([1.0], dtype="float32")
        )
        covars = _CONTINUOUS_COVARIATES + _BINARY_COVARIATES
        covars = covars + _HIDDEN_COVARIATE if not hidden_confounding else covars
        self.dim_input = len(covars)
        self.dim_treatment = 1
        self.dim_output = 1
        if self.split == "test":
            self.x = df_test[covars].to_numpy(dtype="float32")
            self.t = df_test["treat"].to_numpy(dtype="float32")
            self.mu0 = df_test["mu0"].to_numpy(dtype="float32")
            self.mu1 = df_test["mu1"].to_numpy(dtype="float32")
            self.y0 = df_test["y0"].to_numpy(dtype="float32")
            self.y1 = df_test["y1"].to_numpy(dtype="float32")
            if mode == "mu":
                self.y = self.mu1 - self.mu0
            elif mode == "pi":
                self.y = self.t
            else:
                raise NotImplementedError("Not a valid mode")
        else:
            df_train, df_valid = model_selection.train_test_split(
                df_train, test_size=0.3, random_state=seed
            )
            if split == "train":
                df = df_train
            elif split == "valid":
                df = df_valid
            else:
                raise NotImplementedError("Not a valid dataset split")
            self.x = df[covars].to_numpy(dtype="float32")
            self.t = df["treat"].to_numpy(dtype="float32")
            self.mu0 = df["mu0"].to_numpy(dtype="float32")
            self.mu1 = df["mu1"].to_numpy(dtype="float32")
            self.y0 = df["y0"].to_numpy(dtype="float32")
            self.y1 = df["y1"].to_numpy(dtype="float32")
            if mode == "mu":
                self.y = df["y"].to_numpy(dtype="float32")
            elif mode == "pi":
                self.y = self.t
            else:
                raise NotImplementedError("Not a valid mode")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        inputs = (
            torch.from_numpy(self.x[idx]).float()
            if self.mode == "pi"
            else torch.from_numpy(np.hstack([self.x[idx], self.t[idx]])).float()
        )
        targets = torch.from_numpy((self.y[idx] - self.y_mean) / self.y_std).float()
        return inputs, targets
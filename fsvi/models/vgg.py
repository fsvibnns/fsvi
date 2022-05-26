from typing import List, NamedTuple, Union, Type

import jax
import haiku as hk

# TODO: match init from torchvision?
kaiminig_uniform = hk.initializers.VarianceScaling(2.0, "fan_in", "uniform")
# if isinstance(m, nn.Conv2d):
#     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#     if m.bias is not None:
#         nn.init.constant_(m.bias, 0)
# elif isinstance(m, nn.BatchNorm2d):
#     nn.init.constant_(m.weight, 1)
#     nn.init.constant_(m.bias, 0)
# elif isinstance(m, nn.Linear):
#     nn.init.normal_(m.weight, 0, 0.01)
#     nn.init.constant_(m.bias, 0)


class Classifier(hk.Module):
    def __init__(self):
        super().__init__()

        # TODO: Move into __call__ if init is ok?
        self.fc1 = hk.Linear(4096)
        self.fc2 = hk.Linear(4096)
        self.fc3 = hk.Linear(1000)

    def __call__(self, x, is_training):
        dropout_rate = 0.5 if is_training else 0.0

        # torchvision implementation adds an AdaptiveAvgPool2d layer to this point as
        # well to allow the network to work with images larger or smaller than ImageNet.
        # In case of ImageNet the dimension here would be 7 x 7 x 512

        x = self.fc1(x)
        x = jax.nn.relu(x)
        x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
        x = self.fc2(x)
        x = jax.nn.relu(x)
        x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
        x = self.fc3(x)
        return x


class Classifier_Cifar10_Adapted(hk.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = hk.Linear(1024)
        self.fc2 = hk.Linear(512)
        self.fc3 = hk.Linear(10)

    def __call__(self, x, is_training):

        # torchvision implementation adds an AdaptiveAvgPool2d layer to this point as
        # well to allow the network to work with images larger or smaller than ImageNet.
        # In case of ImageNet the dimension here would be 2 x 2 x 512

        x = self.fc1(x)
        x = jax.nn.relu(x)
        x = self.fc2(x)
        x = jax.nn.relu(x)
        x = self.fc3(x)
        return x


class VggConfig(NamedTuple):
    conv_config: List[Union[int, str]]
    classifier: Union[Type[Classifier], Type[Classifier_Cifar10_Adapted]] = Classifier
    include_bn: bool = True


class VGG(hk.Module):
    def __init__(self, config: VggConfig):
        super().__init__()

        self.convs = VGG.make_convs(config.conv_config, config.include_bn)
        self.classifier = config.classifier()

    @staticmethod
    def make_convs(config, include_bn):
        layers = []
        for c in config:
            if c == "M":
                layers.append(
                    hk.MaxPool(
                        window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="VALID"
                    )
                )
            else:
                layers.append(
                    hk.Conv2D(
                        output_channels=c,
                        kernel_shape=3,
                        stride=1,
                        padding="SAME",
                        with_bias=True,
                        data_format="NHWC",
                    )
                )

                if include_bn:
                    layers.append(
                        hk.BatchNorm(
                            create_scale=True,
                            create_offset=True,
                            eps=1e-05,
                            decay_rate=0.9,
                        )
                    )

                layers.append(jax.nn.relu)

        return layers

    def __call__(self, x, is_training):
        for m in self.convs:
            if isinstance(m, hk.BatchNorm):
                x = m(x, is_training)
            else:
                x = m(x)
        x = hk.Flatten()(x)
        x = self.classifier(x, is_training)

        return x


# fmt: off
VGG_CONFIGS = {
    "11_A" : VggConfig([64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]),
    "13_B" : VggConfig([64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]),
    "16_D" : VggConfig([64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]),
    "19_E" : VggConfig([64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]),
    "19_cifar10" : VggConfig([64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512], classifier=Classifier_Cifar10_Adapted)
}
# fmt: on


def vgg(config: str):
    def fwd_fn(x, is_training):
        model = VGG(VGG_CONFIGS[config])
        return model(x, is_training)

    return hk.without_apply_rng(hk.transform_with_state(fwd_fn))
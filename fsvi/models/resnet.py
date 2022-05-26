from typing import Mapping, Optional, Sequence, Union, Any

import haiku as hk
import jax
import jax.numpy as jnp
from jax import random

from haiku import BatchNorm as BatchNorm_reg
from fsvi.utils.haiku_mod import BatchNorm as BatchNorm_mod
from fsvi.utils.haiku_mod import dense_stochastic_hk, conv2D_stochastic
from fsvi.utils.utils import get_inner_layers_stochastic


class BlockV1(hk.Module):
    """ResNet V1 block with optional bottleneck."""

    def __init__(
        self,
        stochastic_parameters: bool,
        batch_normalization: bool,
        batch_normalization_mod: str,
        dropout: bool,
        dropout_rate: float,
        uniform_init_minval: float,
        uniform_init_maxval: float,
        channels: int,
        stride: Union[int, Sequence[int]],
        use_projection: bool,
        bn_config: Mapping[str, float],
        bottleneck: bool,
        name: Optional[str] = None,
        x_condition = None,
    ):

        super().__init__(name=name)
        self.use_projection = use_projection
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.batch_normalization_mod = batch_normalization_mod
        self.x_condition = x_condition
        self.n_condition = self.x_condition.shape[0] if self.x_condition is not None else 0

        bn_config = dict(bn_config)
        bn_config["decay_rate"] = 0.9

        if self.batch_normalization_mod != "not_specified":
            self.BatchNorm = BatchNorm_mod
            if self.batch_normalization_mod == "training_evaluation":
                bn_config["condition_mode"] = "training_evaluation"
                bn_config["n_condition"] = self.n_condition
                bn_config["create_scale"] = False
                bn_config["create_offset"] = False
            elif self.batch_normalization_mod == "evaluation":
                bn_config["condition_mode"] = "evaluation"
                bn_config["n_condition"] = self.n_condition
                bn_config["create_scale"] = False
                bn_config["create_offset"] = False
        else:
            self.BatchNorm = BatchNorm_reg
            bn_config["create_scale"] = True
            bn_config["create_offset"] = True

        if self.use_projection:
            self.proj_conv = conv2D_stochastic(
                output_channels=channels,
                kernel_shape=1,
                stride=stride,
                with_bias=False,
                padding="SAME",
                name="shortcut_conv",
                stochastic_parameters=stochastic_parameters,
                uniform_init_minval=uniform_init_minval,
                uniform_init_maxval=uniform_init_maxval,
            )

            self.proj_batchnorm = self.BatchNorm(name="batchnorm", **bn_config)

        channel_div = 4 if bottleneck else 1
        conv_0 = conv2D_stochastic(
            output_channels=channels // channel_div,
            kernel_shape=1 if bottleneck else 3,
            stride=1,
            with_bias=False,
            padding="SAME",
            name="conv_0",
            stochastic_parameters=stochastic_parameters,
            uniform_init_minval=uniform_init_minval,
            uniform_init_maxval=uniform_init_maxval,
        )

        if self.batch_normalization:
            bn_0 = self.BatchNorm(name="batchnorm", **bn_config)

        conv_1 = conv2D_stochastic(
            output_channels=channels // channel_div,
            kernel_shape=3,
            stride=stride,
            with_bias=False,
            padding="SAME",
            name="conv_1",
            stochastic_parameters=stochastic_parameters,
            uniform_init_minval=uniform_init_minval,
            uniform_init_maxval=uniform_init_maxval,
        )

        if self.batch_normalization:
            bn_1 = self.BatchNorm(name="batchnorm", **bn_config)
            layers = ((conv_0, bn_0), (conv_1, bn_1))
        else:
            layers = ((conv_0), (conv_1))

        if bottleneck:
            conv_2 = conv2D_stochastic(
                output_channels=channels,
                kernel_shape=1,
                stride=1,
                with_bias=False,
                padding="SAME",
                name="conv_2",
                stochastic_parameters=stochastic_parameters,
                uniform_init_minval=uniform_init_minval,
                uniform_init_maxval=uniform_init_maxval,
            )

            if self.batch_normalization:
                scale_init = jnp.zeros if self.batch_normalization_mod == "not_specified" else None
                bn_2 = self.BatchNorm(name="batchnorm", scale_init=scale_init, **bn_config)
                # bn_2 = self.BatchNorm(name="batchnorm", **bn_config)  # TR: removed default scale_init=jnp.zeros
                layers = layers + ((conv_2, bn_2),)
            else:
                layers = layers + ((conv_2))

        self.layers = layers

    def __call__(self, inputs, rng_key, stochastic, is_training, test_local_stats):
        out = shortcut = inputs

        if self.use_projection:
            rng_key, _ = random.split(rng_key)
            shortcut = self.proj_conv(shortcut, rng_key, stochastic)
            if self.batch_normalization:
                shortcut = self.proj_batchnorm(shortcut, is_training, test_local_stats)

        # DROPOUT
        if self.dropout:
            rng_key, _ = random.split(rng_key)
            shortcut = hk.dropout(rng_key, self.dropout_rate, shortcut)

        if self.batch_normalization:
            for i, (conv_i, bn_i) in enumerate(self.layers):
                bn_i.n_condition = self.n_condition

                rng_key, _ = random.split(rng_key)
                out = conv_i(out, rng_key, stochastic)
                out = bn_i(out, is_training, test_local_stats)
                if i < len(self.layers) - 1:  # Don't apply relu or dropout on last layer
                    # DROPOUT
                    if self.dropout:
                        rng_key, _ = random.split(rng_key)
                        out = hk.dropout(rng_key, self.dropout_rate, out)
                    out = jax.nn.relu(out)
        else:
            for i, (conv_i) in enumerate(self.layers):
                rng_key, _ = random.split(rng_key)
                out = conv_i(out, rng_key, stochastic)
                if i < len(self.layers) - 1:  # Don't apply relu or dropout on last layer
                    # DROPOUT
                    if self.dropout:
                        rng_key, _ = random.split(rng_key)
                        out = hk.dropout(rng_key, self.dropout_rate, out)
                    out = jax.nn.relu(out)

        return jax.nn.relu(out + shortcut)


class BlockV2(hk.Module):
    """ResNet V2 block with optional bottleneck."""

    def __init__(
        self,
        stochastic_parameters: bool,
        batch_normalization: bool,
        batch_normalization_mod: str,
        dropout: bool,
        dropout_rate: float,
        uniform_init_minval: float,
        uniform_init_maxval: float,
        channels: int,
        stride: Union[int, Sequence[int]],
        use_projection: bool,
        bn_config: Mapping[str, float],
        bottleneck: bool,
        name: Optional[str] = None,
        x_condition = None,
    ):

        super().__init__(name=name)
        self.use_projection = use_projection
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.batch_normalization_mod = batch_normalization_mod
        self.x_condition = x_condition
        self.n_condition = self.x_condition.shape[0] if self.x_condition is not None else 0

        # TODO: implement dropout
        if self.dropout:
            raise NotImplementedError("Dropout not yet implemented for ResNetV2")

        bn_config = dict(bn_config)
        bn_config["decay_rate"] = 0.9

        if self.batch_normalization_mod != "not_specified":
            self.BatchNorm = BatchNorm_mod
            if self.batch_normalization_mod == "training_evaluation":
                bn_config["condition_mode"] = "training_evaluation"
                bn_config["n_condition"] = self.n_condition
                bn_config["create_scale"] = False
                bn_config["create_offset"] = False
            elif self.batch_normalization_mod == "evaluation":
                bn_config["condition_mode"] = "evaluation"
                bn_config["n_condition"] = self.n_condition
                bn_config["create_scale"] = False
                bn_config["create_offset"] = False
        else:
            self.BatchNorm = BatchNorm_reg
            bn_config["create_scale"] = True
            bn_config["create_offset"] = True

        if self.use_projection:
            self.proj_conv = conv2D_stochastic(
                output_channels=channels,
                kernel_shape=1,
                stride=stride,
                with_bias=False,
                padding="SAME",
                name="shortcut_conv",
                stochastic_parameters=stochastic_parameters,
                uniform_init_minval=uniform_init_minval,
                uniform_init_maxval=uniform_init_maxval,
            )

        channel_div = 4 if bottleneck else 1
        conv_0 = conv2D_stochastic(
            output_channels=channels // channel_div,
            kernel_shape=1 if bottleneck else 3,
            stride=1,
            with_bias=False,
            padding="SAME",
            name="conv_0",
            stochastic_parameters=stochastic_parameters,
            uniform_init_minval=uniform_init_minval,
            uniform_init_maxval=uniform_init_maxval,
        )

        if self.batch_normalization:
            bn_0 = self.BatchNorm(name="batchnorm", **bn_config)

        conv_1 = conv2D_stochastic(
            output_channels=channels // channel_div,
            kernel_shape=3,
            stride=stride,
            with_bias=False,
            padding="SAME",
            name="conv_1",
            stochastic_parameters=stochastic_parameters,
            uniform_init_minval=uniform_init_minval,
            uniform_init_maxval=uniform_init_maxval,
        )

        if self.batch_normalization:
            bn_1 = self.BatchNorm(name="batchnorm", **bn_config)
            layers = ((conv_0, bn_0), (conv_1, bn_1))
        else:
            layers = ((conv_0), (conv_1))

        if bottleneck:
            conv_2 = conv2D_stochastic(
                output_channels=channels,
                kernel_shape=1,
                stride=1,
                with_bias=False,
                padding="SAME",
                name="conv_2",
                stochastic_parameters=stochastic_parameters,
                uniform_init_minval=uniform_init_minval,
                uniform_init_maxval=uniform_init_maxval,
            )

            # NOTE: Some implementations of ResNet50 v2 suggest initializing
            # gamma/scale here to zeros.
            if self.batch_normalization:
                bn_2 = self.BatchNorm(name="batchnorm", **bn_config)
                layers = layers + ((conv_2, bn_2),)
            else:
                layers = layers + ((conv_2),)

        self.layers = layers

    def __call__(self, inputs, rng_key, stochastic, is_training, test_local_stats):
        x = shortcut = inputs

        if self.batch_normalization:
            for i, (conv_i, bn_i) in enumerate(self.layers):
                bn_i.n_condition = self.n_condition

                x = bn_i(x, is_training, test_local_stats)
                x = jax.nn.relu(x)
                if i == 0 and self.use_projection:
                    rng_key, _ = random.split(rng_key)
                    shortcut = self.proj_conv(x, rng_key, stochastic)
                rng_key, _ = random.split(rng_key)
                x = conv_i(x, rng_key, stochastic)
        else:
            for i, (conv_i) in enumerate(self.layers):
                x = jax.nn.relu(x)
                if i == 0 and self.use_projection:
                    rng_key, _ = random.split(rng_key)
                    shortcut = self.proj_conv(x, rng_key, stochastic)
                rng_key, _ = random.split(rng_key)
                x = conv_i(x, rng_key, stochastic)

        return x + shortcut


class BlockGroup(hk.Module):
    """Higher level block for ResNet implementation."""

    def __init__(
        self,
        stochastic_parameters: bool,
        batch_normalization: bool,
        batch_normalization_mod: str,
        dropout: bool,
        dropout_rate: float,
        uniform_init_minval: float,
        uniform_init_maxval: float,
        channels: int,
        num_blocks: int,
        stride: Union[int, Sequence[int]],
        bn_config: Mapping[str, float],
        resnet_v2: bool,
        bottleneck: bool,
        use_projection: bool,
        name: Optional[str] = None,
        x_condition = None,
    ):
        super().__init__(name=name)

        block_cls = BlockV2 if resnet_v2 else BlockV1

        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(
                block_cls(
                    stochastic_parameters=stochastic_parameters,
                    batch_normalization=batch_normalization,
                    batch_normalization_mod=batch_normalization_mod,
                    dropout=dropout,
                    dropout_rate=dropout_rate,
                    uniform_init_minval=uniform_init_minval,
                    uniform_init_maxval=uniform_init_maxval,
                    channels=channels,
                    stride=(1 if i else stride),
                    use_projection=(i == 0 and use_projection),
                    bottleneck=bottleneck,
                    bn_config=bn_config,
                    name="block_%d" % (i),
                    x_condition=x_condition,
                )
            )

    def __call__(self, inputs, rng_key, stochastic, is_training, test_local_stats):
        out = inputs
        for block in self.blocks:
            rng_key, _ = random.split(rng_key)
            out = block(out, rng_key, stochastic, is_training, test_local_stats)
        return out


def check_length(length, value, name):
    if len(value) != length:
        raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


class ResNet(hk.Module):
    """ResNet model."""

    CONFIGS = {
        18: {"blocks_per_group": (2, 2, 2, 2), "bottleneck": False, "channels_per_group": (64, 128, 256, 512),
        "use_projection": (False, True, True, True), },
        34: {"blocks_per_group": (3, 4, 6, 3), "bottleneck": False, "channels_per_group": (64, 128, 256, 512),
            "use_projection": (False, True, True, True), },
        50: {"blocks_per_group": (3, 4, 6, 3), "bottleneck": True, "channels_per_group": (256, 512, 1024, 2048),
            "use_projection": (True, True, True, True), },
        101: {"blocks_per_group": (3, 4, 23, 3), "bottleneck": True, "channels_per_group": (256, 512, 1024, 2048),
            "use_projection": (True, True, True, True), },
        152: {"blocks_per_group": (3, 8, 36, 3), "bottleneck": True, "channels_per_group": (256, 512, 1024, 2048),
            "use_projection": (True, True, True, True), },
        200: {"blocks_per_group": (3, 24, 36, 3), "bottleneck": True, "channels_per_group": (256, 512, 1024, 2048),
            "use_projection": (True, True, True, True), }, }

    BlockGroup = BlockGroup  # pylint: disable=invalid-name
    BlockV1 = BlockV1  # pylint: disable=invalid-name
    BlockV2 = BlockV2  # pylint: disable=invalid-name

    def __init__(
        self,
        no_final_layer_bias: bool,
        stochastic_parameters: bool,
        batch_normalization: bool,
        batch_normalization_mod: str,
        dropout: bool,
        dropout_rate: float,
        extra_linear_layer: bool,
        final_layer_variational: bool,
        fixed_inner_layers_variational_var: bool,
        uniform_init_lin_minval: float,
        uniform_init_lin_maxval: float,
        uniform_init_conv_minval: float,
        uniform_init_conv_maxval: float,
        blocks_per_group: Sequence[int],
        num_classes: int,
        bn_config: Optional[Mapping[str, float]] = None,
        resnet_v2: bool = False,
        bottleneck: bool = True,
        channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
        use_projection: Sequence[bool] = (True, True, True, True),
        logits_config: Optional[Mapping[str, Any]] = None,
        name: Optional[str] = None,
        x_condition = None,
    ):
        """Constructs a ResNet model.
        Args:
        blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
        num_classes: The number of classes to classify the inputs into.
        bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers. By default the
        ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
        resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        ``False``.
        bottleneck: Whether the block should bottleneck or not. Defaults to
        ``True``.
        channels_per_group: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
        use_projection: A sequence of length 4 that indicates whether each
        residual block should use projection.
        logits_config: A dictionary of keyword arguments for the logits layer.
        name: Name of the module.
        """
        super().__init__(name=name)
        self.resnet_v2 = resnet_v2
        self.no_final_layer_bias = no_final_layer_bias
        self.final_layer_variational = final_layer_variational
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.extra_linear_layer = extra_linear_layer
        self.batch_normalization = batch_normalization
        self.batch_normalization_mod = batch_normalization_mod
        self.x_condition = x_condition
        self.n_condition = self.x_condition.shape[0] if self.x_condition is not None else 0

        inner_layers_stochastic = get_inner_layers_stochastic(
            stochastic_parameters=stochastic_parameters,
            final_layer_variational=final_layer_variational,
            fixed_inner_layers_variational_var=fixed_inner_layers_variational_var,
        )

        self.uniform_init_lin_minval = uniform_init_lin_minval
        self.uniform_init_lin_maxval = uniform_init_lin_maxval
        self.uniform_init_conv_minval = uniform_init_conv_minval
        self.uniform_init_conv_maxval = uniform_init_conv_maxval

        bn_config = dict(bn_config or {})
        bn_config.setdefault("eps", 1e-5)
        bn_config["decay_rate"] = 0.9

        if self.batch_normalization_mod != "not_specified":
            self.BatchNorm = BatchNorm_mod
            if self.batch_normalization_mod == "training_evaluation":
                bn_config["condition_mode"] = "training_evaluation"
                bn_config["n_condition"] = self.n_condition
                bn_config["create_scale"] = False
                bn_config["create_offset"] = False
            elif self.batch_normalization_mod == "evaluation":
                bn_config["condition_mode"] = "evaluation"
                bn_config["n_condition"] = self.n_condition
                bn_config["create_scale"] = False
                bn_config["create_offset"] = False
        else:
            self.BatchNorm = BatchNorm_reg
            bn_config["create_scale"] = True
            bn_config["create_offset"] = True

        # TR: added:
        linear_config = dict(logits_config or {})
        # linear_config.setdefault("w_init", jnp.zeros)
        linear_config.setdefault("w_init", None)
        linear_config.setdefault("name", "linear_penultimate")
        linear_config.setdefault("with_bias", True)

        logits_config = dict(logits_config or {})
        # logits_config.setdefault("w_init", jnp.zeros)
        logits_config.setdefault("w_init", None)  # TR: added
        logits_config.setdefault("name", "linear_final")
        logits_config.setdefault("with_bias", not self.no_final_layer_bias)  # TR: originally set to False

        # Number of blocks in each group for ResNet.
        check_length(4, blocks_per_group, "blocks_per_group")
        check_length(4, channels_per_group, "channels_per_group")

        self.initial_conv = conv2D_stochastic(
            output_channels=64,
            kernel_shape=3,
            stride=1,
            with_bias=False,
            padding="SAME",
            name="initial_conv",
            stochastic_parameters=inner_layers_stochastic,
            uniform_init_minval=self.uniform_init_conv_minval,
            uniform_init_maxval=self.uniform_init_conv_maxval,
        )

        if not self.resnet_v2:
            self.initial_batchnorm = self.BatchNorm(name="batchnorm", **bn_config)

        self.block_groups = []
        strides = (1, 2, 2, 2)
        for i in range(4):
            self.block_groups.append(
                BlockGroup(
                    stochastic_parameters=inner_layers_stochastic,
                    batch_normalization=self.batch_normalization,
                    batch_normalization_mod=self.batch_normalization_mod,
                    dropout=self.dropout,
                    dropout_rate=self.dropout_rate,
                    uniform_init_minval=self.uniform_init_conv_minval,
                    uniform_init_maxval=self.uniform_init_conv_maxval,
                    channels=channels_per_group[i],
                    num_blocks=blocks_per_group[i],
                    stride=strides[i],
                    bn_config=bn_config,
                    resnet_v2=resnet_v2,
                    bottleneck=bottleneck,
                    use_projection=use_projection[i],
                    name="block_group_%d" % (i),
                    x_condition=x_condition,
                )
            )

        self.max_pool = hk.MaxPool(
            window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME"
        )

        if self.resnet_v2:
            self.final_batchnorm = self.BatchNorm(name="batchnorm", **bn_config)

        if self.extra_linear_layer:
            self.fc_1 = dense_stochastic_hk(
                output_size=512,
                uniform_init_minval=self.uniform_init_lin_minval,
                uniform_init_maxval=self.uniform_init_lin_maxval,
                stochastic_parameters=inner_layers_stochastic,
                **linear_config,
            )

        self.logits = dense_stochastic_hk(
            output_size=num_classes,
            uniform_init_minval=self.uniform_init_lin_minval,
            uniform_init_maxval=self.uniform_init_lin_maxval,
            stochastic_parameters=stochastic_parameters,
            **logits_config,
        )

    def __call__(self, inputs, rng_key, stochastic, is_training, test_local_stats=False):
        if self.n_condition > 0:
            inputs = jnp.concatenate([inputs, self.x_condition], axis=0)

        self.initial_batchnorm.n_condition = self.n_condition

        out = inputs
        rng_key, _ = random.split(rng_key)
        out = self.initial_conv(out, rng_key, stochastic)
        if not self.resnet_v2:
            if self.batch_normalization:
                out = self.initial_batchnorm(out, is_training, test_local_stats)
            out = jax.nn.relu(out)
            # DROPOUT
            if self.dropout:
                rng_key, _ = random.split(rng_key)
                out = hk.dropout(rng_key, self.dropout_rate, out)

        # out = self.max_pool(out)  # TR: modification made in DUQ

        for block_group in self.block_groups:
            rng_key, _ = random.split(rng_key)
            out = block_group(out, rng_key, stochastic, is_training, test_local_stats)

        if self.resnet_v2:
            out = self.final_batchnorm(out, is_training, test_local_stats)
            out = jax.nn.relu(out)
        out = jnp.mean(out, axis=[1, 2])

        if self.dropout:
            rng_key, _ = random.split(rng_key)
            out = hk.dropout(rng_key, self.dropout_rate, out)

        if self.extra_linear_layer:
            rng_key, _ = random.split(rng_key)
            out = self.fc_1(out, rng_key, stochastic)

            if self.dropout:
                rng_key, _ = random.split(rng_key)
                out = hk.dropout(rng_key, self.dropout_rate, out)

        out = self.logits(out, rng_key, stochastic)

        if self.n_condition > 0:
            out = out[:-self.n_condition]

        return out


class ResNet18(ResNet):
    """ResNet18."""

    def __init__(
        self,
        output_dim: int,
        no_final_layer_bias: bool,
        stochastic_parameters: bool,
        batch_normalization: bool,
        batch_normalization_mod: str,
        dropout: bool,
        dropout_rate: float,
        uniform_init_lin_minval: float,
        uniform_init_lin_maxval: float,
        uniform_init_conv_minval: float,
        uniform_init_conv_maxval: float,
        extra_linear_layer: bool = False,
        final_layer_variational: bool = False,
        fixed_inner_layers_variational_var: bool = False,
        bn_config: Optional[Mapping[str, float]] = None,
        resnet_v2: bool = False,
        logits_config: Optional[Mapping[str, Any]] = None,
        name: Optional[str] = None,
        x_condition = None,
    ):
        """Constructs a ResNet model.

        Args:
          num_classes: The number of classes to classify the inputs into.
          bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
            passed on to the :class:`~haiku.BatchNorm` layers.
          resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
            to ``False``.
          logits_config: A dictionary of keyword arguments for the logits layer.
          name: Name of the module.
        """
        super().__init__(
            no_final_layer_bias = no_final_layer_bias,
            stochastic_parameters=stochastic_parameters,
            batch_normalization=batch_normalization,
            batch_normalization_mod=batch_normalization_mod,
            final_layer_variational=final_layer_variational,
            fixed_inner_layers_variational_var=fixed_inner_layers_variational_var,
            dropout=dropout,
            dropout_rate=dropout_rate,
            extra_linear_layer=extra_linear_layer,
            uniform_init_lin_minval=uniform_init_lin_minval,
            uniform_init_lin_maxval=uniform_init_lin_maxval,
            uniform_init_conv_minval=uniform_init_conv_minval,
            uniform_init_conv_maxval=uniform_init_conv_maxval,
            num_classes=output_dim,
            bn_config=bn_config,
            resnet_v2=resnet_v2,
            **ResNet.CONFIGS[18],
            logits_config=logits_config,
            name=name,
            x_condition=x_condition,
        )

class ResNet34(ResNet):
    """ResNet34."""

    def __init__(
        self,
        output_dim: int,
        no_final_layer_bias: bool,
        stochastic_parameters: bool,
        batch_normalization: bool,
        batch_normalization_mod: str,
        dropout: bool,
        dropout_rate: float,
        uniform_init_lin_minval: float,
        uniform_init_lin_maxval: float,
        uniform_init_conv_minval: float,
        uniform_init_conv_maxval: float,
        extra_linear_layer: bool = False,
        final_layer_variational: bool = False,
        fixed_inner_layers_variational_var: bool = False,
        bn_config: Optional[Mapping[str, float]] = None,
        resnet_v2: bool = False,
        logits_config: Optional[Mapping[str, Any]] = None,
        name: Optional[str] = None,
        x_condition = None,
    ):
        """Constructs a ResNet model.

        Args:
          num_classes: The number of classes to classify the inputs into.
          bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
            passed on to the :class:`~haiku.BatchNorm` layers.
          resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
            to ``False``.
          logits_config: A dictionary of keyword arguments for the logits layer.
          name: Name of the module.
        """
        super().__init__(
            no_final_layer_bias = no_final_layer_bias,
            stochastic_parameters=stochastic_parameters,
            batch_normalization=batch_normalization,
            batch_normalization_mod=batch_normalization_mod,
            final_layer_variational=final_layer_variational,
            fixed_inner_layers_variational_var=fixed_inner_layers_variational_var,
            dropout=dropout,
            dropout_rate=dropout_rate,
            extra_linear_layer=extra_linear_layer,
            uniform_init_lin_minval=uniform_init_lin_minval,
            uniform_init_lin_maxval=uniform_init_lin_maxval,
            uniform_init_conv_minval=uniform_init_conv_minval,
            uniform_init_conv_maxval=uniform_init_conv_maxval,
            num_classes=output_dim,
            bn_config=bn_config,
            resnet_v2=resnet_v2,
            **ResNet.CONFIGS[34],
            logits_config=logits_config,
            name=name,
            x_condition=x_condition,
        )

class ResNet50(ResNet):
    """ResNet50."""

    def __init__(
        self,
        output_dim: int,
        no_final_layer_bias: bool,
        stochastic_parameters: bool,
        batch_normalization: bool,
        batch_normalization_mod: str,
        dropout: bool,
        dropout_rate: float,
        uniform_init_lin_minval: float,
        uniform_init_lin_maxval: float,
        uniform_init_conv_minval: float,
        uniform_init_conv_maxval: float,
        extra_linear_layer: bool = False,
        final_layer_variational: bool = False,
        fixed_inner_layers_variational_var: bool = False,
        bn_config: Optional[Mapping[str, float]] = None,
        resnet_v2: bool = False,
        logits_config: Optional[Mapping[str, Any]] = None,
        name: Optional[str] = None,
        x_condition = None,
    ):
        """Constructs a ResNet model.

        Args:
          num_classes: The number of classes to classify the inputs into.
          bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
            passed on to the :class:`~haiku.BatchNorm` layers.
          resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
            to ``False``.
          logits_config: A dictionary of keyword arguments for the logits layer.
          name: Name of the module.
        """
        super().__init__(
            no_final_layer_bias = no_final_layer_bias,
            stochastic_parameters=stochastic_parameters,
            batch_normalization=batch_normalization,
            batch_normalization_mod=batch_normalization_mod,
            final_layer_variational=final_layer_variational,
            fixed_inner_layers_variational_var=fixed_inner_layers_variational_var,
            dropout=dropout,
            dropout_rate=dropout_rate,
            extra_linear_layer=extra_linear_layer,
            uniform_init_lin_minval=uniform_init_lin_minval,
            uniform_init_lin_maxval=uniform_init_lin_maxval,
            uniform_init_conv_minval=uniform_init_conv_minval,
            uniform_init_conv_maxval=uniform_init_conv_maxval,
            num_classes=output_dim,
            bn_config=bn_config,
            resnet_v2=resnet_v2,
            **ResNet.CONFIGS[50],
            logits_config=logits_config,
            name=name,
            x_condition=x_condition,
        )

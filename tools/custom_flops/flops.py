import sys
import warnings
from functools import partial
from typing import Any, Callable, Dict, Optional, TextIO, Tuple

import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn.utils.flops_counter import add_flops_counting_methods, print_model_with_flops, flops_to_string, params_to_string

def get_model_complexity_info(model: nn.Module,
                              input_data: tuple,
                              print_per_layer_stat: bool = True,
                              as_strings: bool = True,
                              input_constructor: Optional[Callable] = None,
                              flush: bool = False,
                              ost: TextIO = sys.stdout) -> tuple:
    """Get complexity information of a model.

    This method can calculate FLOPs and parameter counts of a model with
    corresponding input shape. It can also print complexity information for
    each layer in a model.

    Supported layers are listed as below:
        - Convolutions: ``nn.Conv1d``, ``nn.Conv2d``, ``nn.Conv3d``.
        - Activations: ``nn.ReLU``, ``nn.PReLU``, ``nn.ELU``,
          ``nn.LeakyReLU``, ``nn.ReLU6``.
        - Poolings: ``nn.MaxPool1d``, ``nn.MaxPool2d``, ``nn.MaxPool3d``,
          ``nn.AvgPool1d``, ``nn.AvgPool2d``, ``nn.AvgPool3d``,
          ``nn.AdaptiveMaxPool1d``, ``nn.AdaptiveMaxPool2d``,
          ``nn.AdaptiveMaxPool3d``, ``nn.AdaptiveAvgPool1d``,
          ``nn.AdaptiveAvgPool2d``, ``nn.AdaptiveAvgPool3d``.
        - BatchNorms: ``nn.BatchNorm1d``, ``nn.BatchNorm2d``,
          ``nn.BatchNorm3d``, ``nn.GroupNorm``, ``nn.InstanceNorm1d``,
          ``InstanceNorm2d``, ``InstanceNorm3d``, ``nn.LayerNorm``.
        - Linear: ``nn.Linear``.
        - Deconvolution: ``nn.ConvTranspose2d``.
        - Upsample: ``nn.Upsample``.

    Args:
        model (nn.Module): The model for complexity calculation.
        input_shape (tuple): Input shape used for calculation.
        print_per_layer_stat (bool): Whether to print complexity information
            for each layer in a model. Default: True.
        as_strings (bool): Output FLOPs and params counts in a string form.
            Default: True.
        input_constructor (None | callable): If specified, it takes a callable
            method that generates input. otherwise, it will generate a random
            tensor with input shape to calculate FLOPs. Default: None.
        flush (bool): same as that in :func:`print`. Default: False.
        ost (stream): same as ``file`` param in :func:`print`.
            Default: sys.stdout.

    Returns:
        tuple[float | str]: If ``as_strings`` is set to True, it will return
        FLOPs and parameter counts in a string format. otherwise, it will
        return those in a float number format.
    """
    # assert type(input_shape) is tuple
    # assert len(input_shape) >= 1
    assert isinstance(model, nn.Module)
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count()
    _ = flops_model(return_loss=False, **input_data)
    # if input_constructor:
    #     input = input_constructor(input_shape)
    #     _ = flops_model(False, **input)
    # else:
    #     try:
    #         batch = torch.ones(()).new_empty(
    #             (1, *input_shape),
    #             dtype=next(flops_model.parameters()).dtype,
    #             device=next(flops_model.parameters()).device)
    #     except StopIteration:
    #         # Avoid StopIteration for models which have no parameters,
    #         # like `nn.Relu()`, `nn.AvgPool2d`, etc.
    #         batch = torch.ones(()).new_empty((1, *input_shape))

    #     _ = flops_model(batch)

    flops_count, params_count = flops_model.compute_average_flops_cost()
    if print_per_layer_stat:
        print_model_with_flops(
            flops_model, flops_count, params_count, ost=ost, flush=flush)
    flops_model.stop_flops_count()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count
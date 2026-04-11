"""Shared utilities for model architectures."""

from typing import Tuple, Optional, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Default variable-to-node grouping for solar wind data (fallback)
# 7 physical variable groups (avg/min/max triplet each) + 1 geomagnetic index
DEFAULT_VARIABLE_NODE_GROUPS = {
    'v': ['v_avg', 'v_min', 'v_max'],
    'np': ['np_avg', 'np_min', 'np_max'],
    't': ['t_avg', 't_min', 't_max'],
    'bx': ['bx_avg', 'bx_min', 'bx_max'],
    'by': ['by_avg', 'by_min', 'by_max'],
    'bz': ['bz_avg', 'bz_min', 'bz_max'],
    'bt': ['bt_avg', 'bt_min', 'bt_max'],
    'ap30': ['ap30'],
}


def build_gnn_node_groups(config):
    """Build GNN node groups from config with validation.

    Reads gnn_variable_groups from config if available, otherwise uses
    DEFAULT_VARIABLE_NODE_GROUPS as fallback. Validates that:
    1. All variables in groups exist in input_variables
    2. All input_variables are assigned to a group
    3. Group order matches input_variables order (sequential split)

    Args:
        config: Hydra config object.

    Returns:
        Tuple of (group_sizes: List[int], num_nodes: int)

    Raises:
        ValueError: If validation fails.
    """
    input_vars = list(config.data.timeseries.input_variables)

    # Get groups from config or fallback
    if hasattr(config.data.timeseries, 'gnn_variable_groups'):
        raw_groups = config.data.timeseries.gnn_variable_groups
        groups = {k: list(v) for k, v in raw_groups.items()}
    else:
        groups = DEFAULT_VARIABLE_NODE_GROUPS

    input_var_set = set(input_vars)

    # Validation 1: all group variables exist in input_variables
    for group_name, var_list in groups.items():
        for var in var_list:
            if var not in input_var_set:
                raise ValueError(
                    f"GNN group '{group_name}' contains variable '{var}' "
                    f"not found in input_variables"
                )

    # Validation 2: all input_variables are in some group
    grouped_vars = set()
    for var_list in groups.values():
        grouped_vars.update(var_list)
    ungrouped = input_var_set - grouped_vars
    if ungrouped:
        raise ValueError(
            f"Input variables {ungrouped} not assigned to any "
            f"gnn_variable_group. Add them to config or update groups."
        )

    # Validation 3: group order matches input_variables sequential order
    expected_order = []
    for var_list in groups.values():
        expected_order.extend(var_list)
    if expected_order != input_vars:
        raise ValueError(
            f"GNN group variable order does not match input_variables order.\n"
            f"  Groups produce: {expected_order}\n"
            f"  Config expects: {input_vars}\n"
            f"Reorder gnn_variable_groups to match input_variables."
        )

    group_sizes = [len(v) for v in groups.values()]
    num_nodes = len(groups)
    return group_sizes, num_nodes


def _get_model_dimensions(config):
    """Compute input/output dimensions from active modality config.

    Returns:
        Tuple of (num_input_variables, input_sequence_length,
                  num_target_variables, target_sequence_length)
    """
    use_csv = getattr(config.data.modalities, 'timeseries', False)

    if use_csv:
        ts_cfg = config.data.timeseries
        num_input_variables = len(ts_cfg.input_variables)
        num_target_variables = len(ts_cfg.target_variables)
        ppd = ts_cfg.points_per_day

        input_start = getattr(ts_cfg, 'input_start', None)
        input_end = getattr(ts_cfg, 'input_end', None)
        target_start = getattr(ts_cfg, 'target_start', None)
        target_end = getattr(ts_cfg, 'target_end', None)

        if input_start is not None and input_end is not None:
            input_sequence_length = input_end - input_start
        else:
            input_sequence_length = ts_cfg.days_before * ppd

        if target_start is not None and target_end is not None:
            target_sequence_length = target_end - target_start
        else:
            target_sequence_length = ts_cfg.days_after * ppd
    else:
        num_input_variables = len(config.data.input_variables)
        input_sequence_length = config.data.input_end_index - config.data.input_start_index
        num_target_variables = len(config.data.target_variables)
        target_sequence_length = config.data.target_end_index - config.data.target_start_index

    # Allow manual override via model.output_seq_len
    if getattr(config.model, 'output_seq_len', None) is not None:
        target_sequence_length = config.model.output_seq_len

    return num_input_variables, input_sequence_length, num_target_variables, target_sequence_length

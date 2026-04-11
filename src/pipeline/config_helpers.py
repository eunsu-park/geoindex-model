"""Config accessor functions for data pipeline.

Provides helper functions for converting between time-based and index-based
configurations, extracting variable lists, and determining data modalities.
"""

import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Time-based Indexing Utilities
# =============================================================================

def hours_to_index(
    hours: int,
    interval: int,
    base_offset: int = -168
) -> int:
    """Convert relative hours to array index.

    Args:
        hours: Relative hours from reference time (e.g., -96 for 4 days before)
        interval: Data interval in hours (e.g., 6 for SDO, 3 for OMNI)
        base_offset: Starting offset of the data in hours (default: -168 = -7 days)

    Returns:
        Array index

    Examples:
        >>> hours_to_index(-96, 6, -168)  # SDO: -96h with 6h interval
        12
        >>> hours_to_index(0, 6, -168)    # SDO: reference time
        28
        >>> hours_to_index(-96, 3, -168)  # OMNI: -96h with 3h interval
        24
        >>> hours_to_index(72, 3, -168)   # OMNI: +72h (3 days)
        80
    """
    return (hours - base_offset) // interval


def days_to_indices_omni(days: List[int], base_offset_hours: int = -168) -> Tuple[int, int]:
    """Convert day list to OMNI array indices.

    OMNI data uses 3-hour intervals (8 points per day).
    Days are numbered: -7 to -1 (input), 1 to 3 (target).
    Day 0 does not exist (reference time is boundary).

    Args:
        days: List of day numbers (negative for input, positive for target)
              Examples: [-7, -6, -5] or [1, 2, 3]
        base_offset_hours: Starting offset in hours (default: -168 = -7 days)

    Returns:
        Tuple of (start_index, end_index) for slicing

    Examples:
        >>> days_to_indices_omni([-7, -6, -5, -4, -3, -2, -1])  # All input days
        (0, 56)
        >>> days_to_indices_omni([1, 2, 3])  # All target days
        (56, 80)
        >>> days_to_indices_omni([-3, -2, -1])  # Last 3 input days
        (32, 56)
        >>> days_to_indices_omni([3])  # Day 3 only
        (72, 80)
    """
    POINTS_PER_DAY = 8  # 24h / 3h = 8

    # Convert days to indices
    # Day -7 -> index 0, Day -1 -> index 48
    # Day +1 -> index 56, Day +3 -> index 72
    indices = []
    for day in sorted(days):
        if day < 0:
            # Input days: day -7 = 0, day -1 = 48
            start_idx = (day + 7) * POINTS_PER_DAY
        elif day > 0:
            # Target days: day +1 = 56, day +3 = 72
            start_idx = 56 + (day - 1) * POINTS_PER_DAY
        else:
            raise ValueError("Day 0 is not valid. Use negative days for input, positive for target.")
        indices.append(start_idx)

    start_index = min(indices)
    # End index is start of last day + points per day
    last_day = sorted(days)[-1]
    if last_day < 0:
        end_index = (last_day + 7 + 1) * POINTS_PER_DAY
    else:
        end_index = 56 + last_day * POINTS_PER_DAY

    return start_index, end_index


def days_to_indices_sdo(days: List[int], base_offset_hours: int = -168) -> Tuple[int, int]:
    """Convert day list to SDO array indices.

    SDO data uses 6-hour intervals (4 points per day).
    Only negative days (input) are valid for SDO.

    Args:
        days: List of negative day numbers (e.g., [-7, -6, -5])
        base_offset_hours: Starting offset in hours (default: -168 = -7 days)

    Returns:
        Tuple of (start_index, end_index) for slicing

    Examples:
        >>> days_to_indices_sdo([-7, -6, -5, -4, -3, -2, -1])  # All 7 days
        (0, 28)
        >>> days_to_indices_sdo([-4, -3, -2, -1])  # Last 4 days
        (12, 28)
        >>> days_to_indices_sdo([-1])  # Last day only
        (24, 28)
    """
    POINTS_PER_DAY = 4  # 24h / 6h = 4

    # Validate: SDO only has input data (negative days)
    if any(d >= 0 for d in days):
        raise ValueError("SDO data only contains input days (negative values)")

    # Convert days to indices
    # Day -7 -> index 0, Day -1 -> index 24
    indices = []
    for day in sorted(days):
        start_idx = (day + 7) * POINTS_PER_DAY
        indices.append(start_idx)

    start_index = min(indices)
    last_day = sorted(days)[-1]
    end_index = (last_day + 7 + 1) * POINTS_PER_DAY

    return start_index, end_index


def get_sdo_indices(config) -> Tuple[int, int]:
    """Get SDO start and end indices from config.

    Supports three configuration methods (in priority order):
    1. Day-based: sampling.input_days list
    2. Time-based: data.sdo.start_hours/end_hours
    3. Legacy index-based: data.sdo_start_index/sdo_end_index

    Args:
        config: Hydra configuration object

    Returns:
        Tuple of (start_index, end_index)
    """
    # Priority 1: Day-based config (sampling.input_days)
    if hasattr(config, 'sampling') and hasattr(config.sampling, 'input_days'):
        input_days = list(config.sampling.input_days)
        return days_to_indices_sdo(input_days)

    # Priority 2: Time-based config (data.sdo.start_hours)
    if hasattr(config.data, 'sdo') and hasattr(config.data.sdo, 'start_hours'):
        interval = config.data.sdo.interval_hours
        base_offset = config.data.sdo.base_offset_hours
        start_idx = hours_to_index(config.data.sdo.start_hours, interval, base_offset)
        end_idx = hours_to_index(config.data.sdo.end_hours, interval, base_offset)
        return start_idx, end_idx

    # Priority 3: Legacy index-based config
    return config.data.sdo_start_index, config.data.sdo_end_index


def get_omni_input_indices(config) -> Tuple[int, int]:
    """Get OMNI input start and end indices from config.

    Supports three configuration methods (in priority order):
    1. Day-based: sampling.input_days list
    2. Time-based: data.omni.input.start_hours/end_hours
    3. Legacy index-based: data.input_start_index/input_end_index

    Args:
        config: Hydra configuration object

    Returns:
        Tuple of (start_index, end_index)
    """
    # Priority 1: Day-based config (sampling.input_days)
    if hasattr(config, 'sampling') and hasattr(config.sampling, 'input_days'):
        input_days = list(config.sampling.input_days)
        return days_to_indices_omni(input_days)

    # Priority 2: Time-based config (data.omni.input.start_hours)
    if hasattr(config.data, 'omni') and hasattr(config.data.omni, 'input'):
        interval = config.data.omni.interval_hours
        base_offset = config.data.omni.base_offset_hours
        start_idx = hours_to_index(config.data.omni.input.start_hours, interval, base_offset)
        end_idx = hours_to_index(config.data.omni.input.end_hours, interval, base_offset)
        return start_idx, end_idx

    # Priority 3: Legacy index-based config
    return config.data.input_start_index, config.data.input_end_index


def get_omni_target_indices(config) -> Tuple[int, int]:
    """Get OMNI target start and end indices from config.

    Supports three configuration methods (in priority order):
    1. Day-based: sampling.target_days list
    2. Time-based: data.omni.target.start_hours/end_hours
    3. Legacy index-based: data.target_start_index/target_end_index

    Args:
        config: Hydra configuration object

    Returns:
        Tuple of (start_index, end_index)
    """
    # Priority 1: Day-based config (sampling.target_days)
    if hasattr(config, 'sampling') and hasattr(config.sampling, 'target_days'):
        target_days = list(config.sampling.target_days)
        return days_to_indices_omni(target_days)

    # Priority 2: Time-based config (data.omni.target.start_hours)
    if hasattr(config.data, 'omni') and hasattr(config.data.omni, 'target'):
        interval = config.data.omni.interval_hours
        base_offset = config.data.omni.base_offset_hours
        start_idx = hours_to_index(config.data.omni.target.start_hours, interval, base_offset)
        end_idx = hours_to_index(config.data.omni.target.end_hours, interval, base_offset)
        return start_idx, end_idx

    # Priority 3: Legacy index-based config
    return config.data.target_start_index, config.data.target_end_index


def get_csv_window_indices(ts_cfg) -> Tuple[int, int, int, int]:
    """Get CSV input/target row indices from timeseries config.

    Supports two configuration methods (in priority order):
    1. Timestep-based: input_start/input_end/target_start/target_end
       (relative to T=0, converted to absolute row indices)
    2. Legacy day-based: days_before/days_after
       (input = [0, days_before*ppd), target = [days_before*ppd, total))

    Args:
        ts_cfg: config.data.timeseries section

    Returns:
        Tuple of (input_start_idx, input_end_idx,
                  target_start_idx, target_end_idx)
    """
    ppd = ts_cfg.points_per_day
    ref_idx = ts_cfg.days_before * ppd  # T=0 position in CSV

    input_start = getattr(ts_cfg, 'input_start', None)
    input_end = getattr(ts_cfg, 'input_end', None)
    target_start = getattr(ts_cfg, 'target_start', None)
    target_end = getattr(ts_cfg, 'target_end', None)

    if input_start is not None and input_end is not None:
        i_s = ref_idx + input_start
        i_e = ref_idx + input_end
    else:
        i_s = 0
        i_e = ref_idx

    if target_start is not None and target_end is not None:
        t_s = ref_idx + target_start
        t_e = ref_idx + target_end
    else:
        t_s = ref_idx
        t_e = ref_idx + ts_cfg.days_after * ppd

    return i_s, i_e, t_s, t_e


def get_sdo_wavelengths(config) -> List[str]:
    """Get SDO wavelengths from config."""
    if hasattr(config.data, 'sdo') and hasattr(config.data.sdo, 'wavelengths'):
        return sorted(list(config.data.sdo.wavelengths))
    return sorted(list(config.data.wavelengths))


def get_input_variables(config) -> List[str]:
    """Get OMNI input variables from config."""
    if hasattr(config.data, 'omni') and hasattr(config.data.omni, 'input'):
        return sorted(list(config.data.omni.input.variables))
    return sorted(list(config.data.input_variables))


def get_target_variables(config) -> List[str]:
    """Get OMNI target variables from config."""
    if hasattr(config.data, 'omni') and hasattr(config.data.omni, 'target'):
        return sorted(list(config.data.omni.target.variables))
    return sorted(list(config.data.target_variables))


def get_sdo_image_size(config) -> int:
    """Get SDO image size from config."""
    if hasattr(config.data, 'sdo') and hasattr(config.data.sdo, 'image_size'):
        return config.data.sdo.image_size
    return config.data.sdo_image_size


def is_timeseries_mode(config) -> bool:
    """Check if CSV timeseries modality is active."""
    return getattr(config.data.modalities, 'timeseries', False)


def get_timeseries_config(config):
    """Get timeseries-specific config section."""
    return config.data.timeseries


def get_timeseries_input_variables(config) -> List[str]:
    """Get input variables for CSV timeseries mode."""
    return list(config.data.timeseries.input_variables)


def get_timeseries_target_variables(config) -> List[str]:
    """Get target variables for CSV timeseries mode."""
    return list(config.data.timeseries.target_variables)


def get_timeseries_normalization_config(config) -> Optional[Dict]:
    """Get normalization config for CSV timeseries mode."""
    ts_cfg = config.data.timeseries
    if hasattr(ts_cfg, 'normalization'):
        norm_config = dict(ts_cfg.normalization)
        if 'methods' in norm_config:
            norm_config['methods'] = dict(norm_config['methods'])
        return norm_config
    return None

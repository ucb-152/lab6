import itertools
import re
import numpy as np

input_channels_list = [128, 256]
output_channels_list = [256]
filter_size_list = [3, 5]
batch_size_list = [4, 16]
image_dims_list = [(32, 32), (256, 256)]
dtype_list = [np.float16, np.float32]
fleet_params = list(itertools.product(
    input_channels_list,
    output_channels_list,   
    filter_size_list,
    batch_size_list,
    image_dims_list,
    dtype_list
))

input_channels_list = [256]
output_channels_list = [256]
filter_size_list = [3]
batch_size_list = [4]
image_dims_list = [(32, 32)]
dtype_list = [np.float32]
basic_params = list(itertools.product(
    input_channels_list,
    output_channels_list,
    filter_size_list,
    batch_size_list,
    image_dims_list,
    dtype_list
))

def params_name(params):
    input_channels, output_channels, filter_size, batch_size, image_dims, dtype = params
    dtype_str = np.zeros(1).astype(dtype).dtype.name
    name = f"in{input_channels}_out{output_channels}_filter{filter_size}x{filter_size}_batch{batch_size}_{image_dims[0]}x{image_dims[0]}_{dtype_str}"
    return name

def test_case_params(test_case):
    if test_case not in test_cases:
        raise ValueError(f"Invalid test case name: {test_case}")

    m = re.match(r"in(\d+)_out(\d+)_filter(\d+)x(\d+)_batch(\d+)_(\d+)x(\d+)_(\w+)", test_case)
    if not m:
        raise ValueError(f"Invalid test case name: {test_case}")
    
    input_channels = int(m.group(1))
    output_channels = int(m.group(2))
    filter_size = int(m.group(3))
    batch_size = int(m.group(5))
    image_dims = (int(m.group(6)), int(m.group(7)))
    dtype_str = m.group(8)
    if dtype_str == "float16":
        dtype = np.float16
    elif dtype_str == "float32":
        dtype = np.float32
    else:
        raise ValueError(f"Invalid dtype: {dtype_str} for test case: {test_case}")

    params = (input_channels, output_channels, filter_size, batch_size, image_dims, dtype)

    if params not in fleet_params:
        raise ValueError(f"Invalid test case params: {params} for test case: {test_case}")

    return params


performance_requirements = {
    params_name(params): 200000 for params in fleet_params
}

dtype_tol = {
    np.float32: {"rtol": 1e-5, "atol": 1e-8},
    np.float16: {"rtol": 1e-3, "atol": 1e-5},
}

test_cases = [
    "in128_out256_filter3x3_batch4_32x32_float16",
    "in128_out256_filter3x3_batch4_32x32_float32",
    "in128_out256_filter3x3_batch4_256x256_float16",
    "in128_out256_filter3x3_batch4_256x256_float32",
    "in128_out256_filter3x3_batch16_32x32_float16",
    "in128_out256_filter3x3_batch16_32x32_float32",
    "in128_out256_filter3x3_batch16_256x256_float16",
    "in128_out256_filter3x3_batch16_256x256_float32",
    "in128_out256_filter5x5_batch4_32x32_float16",
    "in128_out256_filter5x5_batch4_32x32_float32",
    "in128_out256_filter5x5_batch4_256x256_float16",
    "in128_out256_filter5x5_batch4_256x256_float32",
    "in128_out256_filter5x5_batch16_32x32_float16",
    "in128_out256_filter5x5_batch16_32x32_float32",
    "in128_out256_filter5x5_batch16_256x256_float16",
    "in128_out256_filter5x5_batch16_256x256_float32",
    "in256_out256_filter3x3_batch4_32x32_float16",
    "in256_out256_filter3x3_batch4_32x32_float32",
    "in256_out256_filter3x3_batch4_256x256_float16",
    "in256_out256_filter3x3_batch4_256x256_float32",
    "in256_out256_filter3x3_batch16_32x32_float16",
    "in256_out256_filter3x3_batch16_32x32_float32",
    "in256_out256_filter3x3_batch16_256x256_float16",
    "in256_out256_filter3x3_batch16_256x256_float32",
    "in256_out256_filter5x5_batch4_32x32_float16",
    "in256_out256_filter5x5_batch4_32x32_float32",
    "in256_out256_filter5x5_batch4_256x256_float16",
    "in256_out256_filter5x5_batch4_256x256_float32",
    "in256_out256_filter5x5_batch16_32x32_float16",
    "in256_out256_filter5x5_batch16_32x32_float32",
    "in256_out256_filter5x5_batch16_256x256_float16",
    "in256_out256_filter5x5_batch16_256x256_float32",
]
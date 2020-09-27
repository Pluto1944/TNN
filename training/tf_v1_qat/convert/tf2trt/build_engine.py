#!/usr/bin/env python3
# Copyright 2020 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import tensorrt as trt


def build_profile(builder, network, profile_shapes):
    """
    Build optimization profile for the builder and configure the min, opt, max shapes appropriately.
    """

    def is_dimension_dynamic(dim):
        return dim is None or dim <= 0

    def override_shape(shape):
        return tuple([1 if is_dimension_dynamic(dim) else dim for dim in shape])

    profile = builder.create_optimization_profile()
    for idx in range(network.num_inputs):
        inp = network.get_input(idx)

        def get_profile_shape(name):
            if name not in profile_shapes:
                return None
            shapes = profile_shapes[name]
            return shapes

        if inp.is_shape_tensor:
            shapes = get_profile_shape(inp.name)
            min, opt, max = shapes
            profile.set_shape_input(inp.name, min, opt, max)
            print("Setting shape input: {:} values to min: {:}, opt: {:}, max: {:}".format(inp.name, min, opt, max))
        elif -1 in inp.shape:
            shapes = get_profile_shape(inp.name)
            if not shapes:
                shapes = [override_shape(inp.shape)] * 3
                print(
                    "Overriding dynamic input shape {:} to {:}. If this is incorrect, "
                    "for input tensor: {:}, please provide tuples for min, opt, "
                    "and max shapes containing values: {:} with dynamic dimensions replaced,".format(
                        inp.shape, shapes[0], inp.name, inp.shape))
            min, opt, max = shapes
            profile.set_shape(inp.name, min, opt, max)
            print("Setting input: {:} shape to min: {:}, opt: {:}, max: {:}".format(inp.name, min, opt, max))
    if not profile:
        print("Profile is not valid, please provide profile data. Note: profile was: {:}".format(profile_shapes))
    return profile


def build_engine_onnx(model_file, verbose=False):
    """
    Parse the model file through TensorRT, build TRT engine and run inference
    """
    # Create builder and network
    if verbose:
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    else:
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network_flags = network_flags | (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION))

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags=network_flags) as network, trt.OnnxParser(
            network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30
        config.flags = config.flags | 1 << int(trt.BuilderFlag.INT8)
        # Setting the (min, opt, max) batch sizes to be 1. Users need to configure this according to their requirements.
        config.add_optimization_profile(build_profile(builder, network, profile_shapes={
            'input': [(1, 3, 224, 224), (1, 3, 224, 224), (1, 3, 224, 224)]}))

        return builder.build_engine(network, config)


def main(args):
    model_file = args.onnx
    # Parse the ONNX graph through TensorRT and build the engine
    trt_engine = build_engine_onnx(model_file, args.verbose)
    # Serialize the engine and save to file
    with open(args.engine, "wb") as file:
        file.write(trt_engine.serialize())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default='rn50.onnx', help="Path to RN50 ONNX graph")
    parser.add_argument("--engine", type=str, default='rn50_trt.engine', help="output path to TensorRT engine")
    parser.add_argument('-v', '--verbose', action='store_true', help="Flag to enable verbose logging")
    args = parser.parse_args()
    main(args)
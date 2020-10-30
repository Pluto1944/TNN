# Tensorflow QAT models to TensorRT

## Description
This sample demonstrates workflow for inferencing a Resnet-50 model trained using Quantization Aware Training with TensorRT.

## Prerequisites
 - TensorRT >=7.1
 - Tensorflow=1.15
 - cuda >=10.2
 - pycuda
 - pillow

## Running the sample

### Step 1. Postprocessing checkpoint
 - Loads a RN50 checkpoint with Dense layer as the final layer and transforms the final dense layer into a 1x1 convolution layer. 
    ```
    cd training/tf_v1_qat/convert/tf2trt
    python postprocess_checkpoint.py --input /path/model.ckpt --dense_layer resnet_model/dense/kernel
    ```

### Step 2. Freeze graph to pb
 - Freeze tensorflow graph to pb. The input_format and the compute_format must be NCHW.
    ```
    cd training/tf_v1_qat/convert/tf2trt
    python export_frozen_graph.py --checkpoint=/path/new.ckpt --quantize --input_format NCHW --output_file=/path/resnet50v1.5.pb --symmetric --use_qdq --use_final_conv
    ```

### Step 3. Tensorflow pb to ONNX
 - Convert tensorflow pb to ONNX. This step needs modifications in Tensorflow graph grappler to disable quantization scales folding into convolution op. You can use <a href="https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow">TensorFlow NGC containers</a> or compile Tensorflow from source with this <a href="https://github.com/NVIDIA/tensorflow/commit/56d0fcb3ebc72c64deeed11e3443dae0a6bbee01#diff-f3c968af33d813270dacefabc01b6a73">commit</a>.
    ```
    python3 -m tf2onnx.convert --input /path/resnet50v1.5.pb --output /path/resnet50v1.5.onnx --inputs input:0 --outputs softmax_tensor:0 --opset 11
    ```

### Step 4. ONNX postprocess
- Postprocess onnx to remove some transpose nodes.
    ```
    cd training/tf_v1_qat/convert/tf2trt
    python postprocess_onnx.py --input /path/resnet50v1.5.onnx --output /path/postprocessed_rn50.onnx
    ```
   
### Step 5. Build TensorRT engine
- Build TensorRT engine using python API.
    ```
    cd training/tf_v1_qat/convert/tf2trt
    python build_engine.py --onnx /path/postprocessed_rn50.onnx --engine /path/rn50qat.trt
    ```

### Step 6. Validation
- Validation on a single image using python API.
    ```
    cd training/tf_v1_qat/convert/tf2trt
    python infer.py --engine /path/rn50qat.trt --image ILSVRC2012_val_00034812.JPEG
    ```
- The output is like this
    ```
    Output class of the image: marmot Confidence: 0.6862110495567322
    ```

## Roadmap
- Per-channel quantization
- Per-layer quantization
- Supporting more models
- ...

## References
- Nvidia TensorRT <a href="https://github.com/NVIDIA/sampleQAT">sampleQAT</a> tools
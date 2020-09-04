# Tensorflow QAT models to TensorRT

## Description
This sample demonstrates workflow for inferencing a Resnet-50 model trained using Quantization Aware Training with TensorRT.

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
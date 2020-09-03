# Resnet-50 trained with QAT

## Description
This sample demonstrates workflow for training and inference of Resnet-50 model trained using Quantization Aware Training. The inference supports TNN and TensorRT.

## Running the sample

### Step 1. Train a model in fp32
 - Dependencies required for this sample: Tensorflow 1.15

    ```
    cd training/tf_v1_qat/examples/resnet
    python imagenet_main.py --data_dir=/path/imagenet/train --num_gpus=4 --rv=1
    ```

### Step 2. Quantization Aware Training
  - Finetune a RN50 model with quantization nodes and save the final checkpoint.

    ```
    cd training/tf_v1_qat/examples/resnet
    python imagenet_main.py --data_dir=/path/imagenet/train --num_gpus=1 --rv=1 --pretrained_model_checkpoint_path=/ckptpath/ --tf_quant --bs=64
    ```
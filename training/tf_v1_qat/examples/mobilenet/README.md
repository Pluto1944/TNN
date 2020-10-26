# Mobilenet-v2 trained with QAT

## Description
This sample demonstrates workflow for training and inference of Mobilenet-v2 model trained using Quantization Aware Training. The inference supports TNN and TensorRT.

## Running the sample

### Step 1. Train a model in fp32
 - Dependencies required for this sample: Tensorflow 1.15

    ```
    cd training/tf_v1_qat/examples/
    python imagenet_main.py --data_dir=/path/imagenet/train \
      --model_name=mobilenetv2 --num_gpus=8 --weight_decay=4e-5 \
      --label_smoothing=0.1 --bs=768 --train_epochs=450
    ```

### Step 2. Quantization Aware Training
  - Finetune a Mobilenet-v2 model with quantization nodes and save the final checkpoint.

    ```
    cd training/tf_v1_qat/examples/
    python imagenet_main.py --data_dir=/path/imagenet/train --model_name=mobilenetv2 --num_gpus=1 \
    --pretrained_model_checkpoint_path=/ckptpath/ --tf_quant --bs=64
    ```
# Quantization-aware training examples


## Quantized accuracy results
The following are results of training some popular CNN models (Resnet50-v1) using this tool:

<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Top-1 Accuracy:<br>fp32</th>
      <th>Top-1 Accuracy:<br>int8 qat</th>
      <th>Top-1 Accuracy:<br>int8 post-train</th>
    </tr>
    <tr><td>Resnet50-v1</td><td>0.7654</td><td>0.7673</td><td>0.7617</td></tr>
  </table>
  <figcaption>
    <b>Table 1</b>: Top-1 accuracy of floating point and fully quantized CNNs on Imagenet Validation dataset.
  </figcaption>
</figure>

## Quantized model speed in NVIDIA T4 with TensorRT
<figure>
  <table>
    <tr>
      <th>Model</th>
      <th>Host latency: <br>fp32</th>
      <th>Host latency: <br>int8 qat</th>
      <th>Host latency: <br>int8 post-train</th>
    </tr>
    <tr><td>Resnet50-v1(batchsize=1)</td><td>3.46ms</td><td>1.03ms</td><td>0.94ms</td></tr>
    <tr><td>Resnet50-v1(batchsize=64)</td><td>125.59ms</td><td>20.06ms</td><td>18.27ms</td></tr>
  </table>
  <figcaption>
    <b>Table 2</b>: Host latency of floating point and fully quantized Resnet50-v1
  </figcaption>
</figure>
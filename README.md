# Edge TPU Multiple Video Streams Demo

This repo contains an end to end demo source code that allows to process
multiple video streams with ML inference using 8 EdgeTPUs in one system
together with the [TensorFlow Lite API](https://tensorflow.org/lite) with a
Coral devices such as the
[USB Accelerator](https://coral.withgoogle.com/products/accelerator) or
[Dev Board](https://coral.withgoogle.com/products/dev-board) and provides an
Object tracker for use with the detected objects.

The demo is hardcoded to run 5 separate video streams (defined in the Pipeline
class) with different
types of inferenceing running on the streams on a total number of 8 TPUs,
and requires an ASUS AI Accelerator PCIe Card. Since the gstreamer pipeline
is pretty demanding, a mid to high end GPU is recommended.

The window shows 6 video windows as follows:

1. Cars on highway with pipelined inferencing, 4 TPUs
1. Workers walking around with segmentation inferencing, 1 TPU
1. Workers walking around with detection inferencing and a keepout zone, 1
TPU
1. Back yard video with detection inferencing and identification, 1 TPU
1. Bird video with object detection marking up birds and cropping the next
window, 0.5 TPU
1. Cropped window of the same bird video as previous with classification of
species (This model is cocompiled with the previous bird detection model), 0.5
TPU

Pressing 1-6 brings the selected video to the front and scales it up.
Pressing any other key brings back the tiled view.

## Building the demo for x86
The demo is only tested on an x86

If you have docker installed you can build it by typing:

```
make DOCKER_TARGETS=demo docker-build
```
building it locally is done by:

```
make demo
```

## Running the demo

```
./MultiVideoStreamsDemo
```
That's all!

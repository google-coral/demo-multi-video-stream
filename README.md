# Edge TPU Multiple Video Streams Demo

This repo contains an end to end demo source code that allows to process
multiple video streams with ML inference using 8 EdgeTPUs in one system.
It is intended to be used with the [ASUS AI Accelerator PCIe Card](https://iot.asus.com/products/AI-accelerator/AI-Accelerator-PCIe-Card/) 
although it is also possible to use x86 or ARM systems with 8 TPUs (likely 
combining multiple [USB Accelerators](https://coral.withgoogle.com/products/accelerator) and a 
host like the the [Dev Board](https://coral.withgoogle.com/products/dev-board)).

The demo is hardcoded to run 5 separate video streams (defined in the Pipeline
class) with different types of inferenceing running on the streams on a total 
number of 8 TPUs. In addition to the 8 TPUs, a mid to high end GPU is 
recommended to handle the mixing/rendering done by the GStreamer pipeline.

The window shows 6 video windows as follows:

1. Cars on highway with pipelined inferencing, 4 TPUs
1. Students in classroom with people segmentation, 1 TPU
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

If you have docker installed you can build it by typing (preferred way):

```
make DOCKER_TARGETS=demo docker-build
```
It is also possible to build locally, but you should validate your system has
the dependecies listed in the [Dockerfile](docker/Dockerfile).

```
make demo
```

## Running the demo

```
./MultiVideoStreamsDemo
```
That's all!

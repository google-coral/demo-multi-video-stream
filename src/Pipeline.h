/*
 * Copyright 2021 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * Pipeline.h
 *
 *  Created on: Mar 12, 2021
 *      Author: pnordstrom
 */

#ifndef PIPELINE_H_
#define PIPELINE_H_

#include <vector>

#include "InferencerBin.h"

namespace szd {

class Pipeline {
 public:
  Pipeline(int argc, char **argv);
  Pipeline() = delete;
  virtual ~Pipeline();

  void Run();

 private:
  std::vector<std::string> kVideoStreams = { "videos/video_device.mp4",
      "videos/classroom.mp4", "videos/worker-zone-detection.mp4",
      "videos/garden.mp4", "videos/birds.mp4", "videos/birds.mp4" };
  const char *kPipelinedModel = "models/efficientdet_lite3_512_ptq";
  const char *kPipelinedLabels = "models/coco_labels.txt";
  const char *kPipelinedObject = "car";
  const size_t kPipelinedNumTPUs = 4;

  const char *kSegmentationModel =
      "models/deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite";
  const char *kSegmentationLabels = "models/deeplab_labels.txt";
  const char *kSegmentationObject = "person";

  const char *kManufacturingModel =
      "models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite";
  const char *kManufacturingLabels = "models/coco_labels.txt";
  const char *kManufacuringPolygon = "models/keepout_points.csv";

  const char *kCoCompiledModel1 =
      "models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite";
  const char *kCoCompiledLabels1 = "models/coco_labels.txt";
  const char *kCoCompiledObject = "bird";

  const char *kCoCompiledModel2 =
      "models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite";
  const char *kCoCompiledLabels2 = "models/birds_labels.txt";

  const char *kDetectionModel =
      "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite";
  const char *kDetectionLabels = "models/coco_labels.txt";

  const float kThreshold = 0.5;
  const char *kAnyObject = "all";
  std::vector<std::shared_ptr<InferencerBin>> inferencer_bins_;
  std::shared_ptr<MixerBin> mixer_;

  struct user_data {
    GMainLoop *loop;
  };

  static gboolean BusWatcher(GstBus *bus, GstMessage *msg, gpointer data);

  GstBus *bus_;
  GMainLoop *loop_;
  GstElement *pipeline_;
  guint bus_watch_id_;
  user_data ud_;
};

} /* namespace szd */

#endif /* PIPELINE_H_ */

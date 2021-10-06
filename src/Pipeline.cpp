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
 * Pipeline.cpp
 *
 *  Created on: Mar 12, 2021
 *      Author: pnordstrom
 */

#ifndef NO_INFERENCING
#define NO_INFERENCING 0  // Set to 1 to only run the Gstreamer code, useful when no TPUs available
#endif

#include <functional>
#include <string>

#include <glib.h>
#include <gst/gst.h>

#include "ClassificationInferencer.h"
#include "DetectionInferencer.h"
#include "InferencerBin.h"
#include "ManufacturingInferencer.h"
#include "MixerBin.h"
#include "Pipeline.h"
#include "PipelinedInferencer.h"
#include "SegmentationInferencer.h"
#include "TwoModelInferencerBin.h"

namespace szd {

gboolean Pipeline::BusWatcher(GstBus *bus, GstMessage *msg, gpointer data) {
  auto loop = reinterpret_cast<Pipeline::user_data*>(data)->loop;

  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
      g_main_loop_quit(loop);
      break;

    case GST_MESSAGE_ERROR: {
      gchar *debug;
      GError *error;

      gst_message_parse_error(msg, &error, &debug);
      g_free(debug);

      g_printerr("%s\n", error->message);
      g_error_free(error);

      g_main_loop_quit(loop);
      break;
    }
    default:
      break;
  }

  return TRUE;
}

Pipeline::Pipeline(int argc, char **argv) {
  gst_init(&argc, &argv);
  pipeline_ = gst_pipeline_new("video-player");
  loop_ = g_main_loop_new(NULL, FALSE);
  bus_ = gst_pipeline_get_bus(GST_PIPELINE(pipeline_));
  ud_ = { loop_ };
  bus_watch_id_ = gst_bus_add_watch(bus_, BusWatcher,
                                    reinterpret_cast<void*>(&ud_));
  mixer_ = std::make_shared<MixerBin>();
  gst_object_unref(bus_);

#if NO_INFERENCING
  auto piplined_inferencer = std::make_shared<InferencerBase>();
  auto seg_inferencer = std::make_shared<InferencerBase>();
  auto mfg_inferencer = std::make_shared<InferencerBase>();
  auto det_to_class_inferencer = std::make_shared<InferencerBase>();
  auto class_inferencer = std::make_shared<InferencerBase>();
  auto det_inferencer_mnv2_2 = std::make_shared<InferencerBase>();

#else
  auto piplined_inferencer = std::make_shared<PipelinedInferencer>(
      kPipelinedModel, kPipelinedLabels, kPipelinedObject, kThreshold,
      kPipelinedNumTPUs);
  auto seg_inferencer = std::make_shared<SegmentationInferencer>(
      kSegmentationModel, kSegmentationLabels, kSegmentationObject, kThreshold);
  auto mfg_inferencer = std::make_shared<ManufacturingInferencer>(
      kManufacturingModel, kManufacturingLabels, kThreshold,
      kManufacuringPolygon);
  auto det_to_class_inferencer = std::make_shared<DetectionInferencer>(
      kCoCompiledModel1, kCoCompiledLabels1, kThreshold, kCoCompiledObject);
  auto class_inferencer = std::make_shared<ClassificationInferencer>(
      kCoCompiledModel2, kCoCompiledLabels2, kThreshold,
      *det_to_class_inferencer);
  auto det_inferencer_mnv2_2 = std::make_shared<DetectionInferencer>(
      kDetectionModel, kDetectionLabels, kThreshold, kAnyObject);
#endif

  // Put together the Gstreamer Pipeline
  gst_bin_add(GST_BIN(pipeline_), mixer_->GetBin());
  int i = 0;
  std::vector<std::shared_ptr<InferencerBase>> inferencers =
      { piplined_inferencer, seg_inferencer, mfg_inferencer,
          det_inferencer_mnv2_2 };
  for (auto inferencer : inferencers) {
    inferencer_bins_.push_back(
        std::make_shared<InferencerBin>(inferencer, kVideoStreams[i]));
    i += 1;
  }

  inferencer_bins_.push_back(
      std::make_shared<TwoModelInferencerBin>(det_to_class_inferencer,
                                              class_inferencer,
                                              kVideoStreams[i++]));
  for (auto infbin : inferencer_bins_) {
    CHECK(gst_bin_add(GST_BIN(pipeline_),infbin->GetBin()));
    CHECK(mixer_->LinkInput(*infbin));
  }

}

Pipeline::~Pipeline() {
}

void Pipeline::Run() {

  GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN (pipeline_),
                                    GST_DEBUG_GRAPH_SHOW_ALL,
                                    "myplayer_before_play");
  gst_element_set_state(pipeline_, GST_STATE_PLAYING);

  // Wait for pipeline to reach PLAYING
  gst_element_get_state(pipeline_, NULL, NULL, GST_CLOCK_TIME_NONE);

  for (auto infbin : inferencer_bins_) {
    infbin->Rewind();
  }

  GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN (pipeline_),
                                    GST_DEBUG_GRAPH_SHOW_ALL,
                                    "myplayer_after_play");

  g_main_loop_run(loop_);

  gst_element_set_state(pipeline_, GST_STATE_NULL);

  gst_object_unref(GST_OBJECT(pipeline_));
  g_source_remove(bus_watch_id_);
  g_main_loop_unref(loop_);

  return;
}

} /* namespace szd */

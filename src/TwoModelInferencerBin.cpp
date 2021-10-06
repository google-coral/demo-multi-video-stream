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
 * TwoModelInferencerBin.cpp
 *
 *  Created on: Sep 2, 2021
 *      Author: pnordstrom
 */

#include <vector>

#include <gst/video/video.h>

#include "absl/strings/substitute.h"
#include "TwoModelInferencerBin.h"
#include "ClassificationInferencer.h"

namespace szd {

static bool score_compare(DetectionResult a, DetectionResult b) {
  return (a.score < b.score);
}

GstFlowReturn TwoModelInferencerBin::AppsinkOnNewSample(GstElement *sink) {
  GstSample *sample = NULL;
  GstFlowReturn retval = GST_FLOW_OK;
  std::shared_ptr<void> output_data;

  switch (auto type = inferencer_->GetInferencerType()) {
    case kDetection:
    case kClassification:
      g_signal_emit_by_name(sink, "pull-sample", &sample);
      if (!sample) {
        g_error("Failed to pull appsink sample\n");
        return GST_FLOW_ERROR;
      }

      if (mixer_->DoInterpret(this)) {
        GstMapInfo info;
        auto buf = gst_sample_get_buffer(sample);
        if (gst_buffer_map(buf, &info, GST_MAP_READ) == TRUE) {
          auto meta = gst_buffer_get_video_meta(buf);
          // Pass the frame to the inferencer
          if (sink == appsink_0_) {
            inferencer_->InterpretFrame(info.data, info.size, meta->width,
                                        meta->height, meta->stride[0],
                                        output_data);
            auto results =
                *std::static_pointer_cast<std::vector<DetectionResult>>(
                    output_data);
            if (!results.empty()) {
              if (results.size() > 1) {
                auto result = std::max_element(results.begin(), results.end(),
                                               score_compare);
                results = {*result};
              }

              int crop_left = results[0].x1 * cropper_input_width_;
              int crop_right =
                  cropper_input_width_ - results[0].x2 * cropper_input_width_;
              int crop_top = results[0].y1 * cropper_input_height_;
              int crop_bottom =
                  cropper_input_height_ - results[0].y2 * cropper_input_height_;
              g_object_set(G_OBJECT(cropper_), "left", crop_left, "right",
                           crop_right, "top", crop_top, "bottom", crop_bottom,
                           NULL);

              auto svg_string = ResultsToSvg(results);
              OutputInferenceResult(svg_string);
            } else {
              g_object_set(G_OBJECT(cropper_), "left", 0, "right", 0, "top", 0,
                           "bottom", 0, NULL);
              OutputInferenceResult("");
            }
          } else if (sink == appsink_1_) {
            std::string output;
            second_inferencer_->InterpretFrame(info.data, info.size,
                                               meta->width, meta->height,
                                               meta->stride[0], output_data);
            auto results =
                *std::static_pointer_cast<std::vector<ClassificationResult>>(
                    output_data);
            if (!results.empty()) {
              output = absl::StrCat(second_inferencer_->GetModelDescription(),
                                    "\n", results[0].candidate);
            } else {
              output = second_inferencer_->GetModelDescription();
            }
            g_object_set(G_OBJECT(text_overlay_1_), "text", output.c_str(),
                         NULL);
          }
          gst_buffer_unmap(buf, &info);
        } else {
          g_error("Couldn't map buffer\n");
          retval = GST_FLOW_ERROR;
        }
      }
      gst_sample_unref(sample);
      break;
    case kNone:
      g_signal_emit_by_name(sink, "pull-sample", &sample);
      gst_sample_unref(sample);
      break;
    default:
      g_error("Unsupported inferencer type %d for %s\n", type, __FILE__);
      exit(1);
  }
  return retval;
}

void TwoModelInferencerBin::SetFullScreenCaps(int src_pad) {
  std::string caps = "video/x-raw, width=$0, height=$1, pixel-aspect-ratio=1/1";
  caps = absl::Substitute(caps, fullscreen_video_width_,
                          fullscreen_video_height_);

  auto filter = src_pad == 0 ? filter_0_ : filter_1_;
  gst_util_set_object_arg(G_OBJECT(filter), "caps", caps.c_str());
}

void TwoModelInferencerBin::SetTiledViewCaps(int src_pad) {
  std::string caps = "video/x-raw, width=$0, height=$1, pixel-aspect-ratio=1/1";
  caps = absl::Substitute(caps, tiled_video_width_, tiled_video_height_);

  auto filter = src_pad == 0 ? filter_0_ : filter_1_;
  gst_util_set_object_arg(G_OBJECT(filter), "caps", caps.c_str());
}

GstPadProbeReturn TwoModelInferencerBin::CropperSinkPadCallback(
    GstPad *pad, GstPadProbeInfo *info) {
  auto event = gst_pad_probe_info_get_event(info);
  if (GST_EVENT_TYPE(event) == GST_EVENT_CAPS) {
    GstCaps *caps = gst_caps_new_any();
    gst_event_parse_caps(event, &caps);

    GstStructure *s = gst_caps_get_structure(caps, 0);

    auto res = gst_structure_get_int(s, "width", &cropper_input_width_);
    CHECK(res);
    res |= gst_structure_get_int(s, "height", &cropper_input_height_);
    CHECK(res);
  } else if (GST_EVENT_TYPE(event) == GST_EVENT_RECONFIGURE) {
    return GST_PAD_PROBE_DROP;
  }

  return GST_PAD_PROBE_OK;
}

TwoModelInferencerBin::TwoModelInferencerBin(
    std::shared_ptr<InferencerBase> first_inferencer,
    std::shared_ptr<InferencerBase> second_inferencer, std::string video_file)
    :
    InferencerBin(first_inferencer),
    second_inferencer_(second_inferencer) {

  SetupAllDims(video_file);

  auto bin_src = absl::Substitute(inferencer_bin_src_, tiled_video_width_,
                                  tiled_video_height_,
                                  inferencer_->GetInputWidth(),
                                  inferencer_->GetInputHeight(),
                                  second_inferencer_->GetInputWidth(),
                                  second_inferencer_->GetInputHeight());
  SetupBin(bin_src, video_file);

  filter_1_ = gst_bin_get_by_name(GST_BIN(bin_), "filter_1");

  appsink_0_ = gst_bin_get_by_name(GST_BIN(bin_), "appsink_0");
  g_object_set(appsink_0_, "emit-signals", true, NULL);
  g_signal_connect(
      appsink_0_,
      "new-sample",
      reinterpret_cast<GCallback>(+[](
          GstElement *sink, TwoModelInferencerBin *self) -> GstFlowReturn {
        return self->AppsinkOnNewSample(sink);
      }),
      this);

  appsink_1_ = gst_bin_get_by_name(GST_BIN(bin_), "appsink_1");
  g_object_set(appsink_1_, "emit-signals", true, NULL);
  g_signal_connect(
      appsink_1_,
      "new-sample",
      reinterpret_cast<GCallback>(+[](
          GstElement *sink, TwoModelInferencerBin *self) -> GstFlowReturn {
        return self->AppsinkOnNewSample(sink);
      }),
      this);

  // set up callback for cropper sink pad to query width and height
  cropper_ = gst_bin_get_by_name(GST_BIN(bin_), "cropper");
  auto cropper_sink_pad = gst_element_get_static_pad(cropper_, "sink");
  gst_pad_add_probe(
      cropper_sink_pad,
      GST_PAD_PROBE_TYPE_EVENT_BOTH,
      reinterpret_cast<GstPadProbeCallback>(+[](
          GstPad *pad, GstPadProbeInfo *info,
          TwoModelInferencerBin *self) -> GstPadProbeReturn {
        return self->CropperSinkPadCallback(pad, info);
      }),
      this, NULL);

  text_overlay_1_ = gst_bin_get_by_name(GST_BIN(bin_), "text_1");
  g_object_set(G_OBJECT(text_overlay_1_), "text",
               second_inferencer_->GetModelDescription().c_str(), NULL);

  // Setup the output pads to be connected to the mixer
  auto source_pad_internal = gst_element_get_static_pad(text_overlay_0_, "src");
  auto source_pad = gst_ghost_pad_new("inf_bin_src_0", source_pad_internal);
  gst_element_add_pad(bin_, source_pad);
  gst_object_unref(source_pad_internal);

  source_pad_internal = gst_element_get_static_pad(text_overlay_1_, "src");
  source_pad = gst_ghost_pad_new("inf_bin_src_1", source_pad_internal);
  gst_element_add_pad(bin_, source_pad);
  gst_object_unref(source_pad_internal);

  num_src_pads_ = 2;

}

TwoModelInferencerBin::~TwoModelInferencerBin() {
  gst_object_unref(filter_1_);
  gst_object_unref(appsink_0_);
  gst_object_unref(appsink_1_);
  gst_object_unref(text_overlay_1_);
  gst_object_unref(cropper_);
}

}  /* namespace szd */

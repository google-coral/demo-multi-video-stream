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
 * MixerBin.cpp
 *
 *  Created on: Mar 18, 2021
 *      Author: pnordstrom
 */

#include <array>
#include <string>
#include <typeindex>
#include <typeinfo>

#include <glib.h>
#include <gst/gst.h>

#include "MixerBin.h"
#include "TwoModelInferencerBin.h"

namespace szd {
void MixerBin::TiledView() {
  if (fullscreen_) {
    linked_inputs_[fullscreen_stream_].inputstream->SetTiledViewCaps(
        linked_inputs_[fullscreen_stream_].src_pad_no_);
  }
  fullscreen_ = false;
  fullscreen_stream_ = -1;
}

void MixerBin::FullScreen(int stream_no) {
  if (fullscreen_stream_ == stream_no)
    return;

  linked_inputs_[stream_no].inputstream->SetFullScreenCaps(
      linked_inputs_[stream_no].src_pad_no_);
  if (fullscreen_) {
    linked_inputs_[fullscreen_stream_].inputstream->SetTiledViewCaps(
        linked_inputs_[fullscreen_stream_].src_pad_no_);
  }
  fullscreen_ = true;
  fullscreen_stream_ = stream_no;
}

bool MixerBin::DoInterpret(InferencerBin *stream) {
  return !fullscreen_
      || linked_inputs_[fullscreen_stream_].inputstream == stream;
}

GstPadProbeReturn MixerBin::SinkPadCallback(GstPad *pad,
                                            GstPadProbeInfo *info) {
  auto event = gst_pad_probe_info_get_event(info);
  switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_CAPS: {
      for (auto record : linked_inputs_) {
        if (pad == record.sink_pad_external_) {
          if (fullscreen_ && fullscreen_stream_ == record.stream_no_) {
            g_object_set(G_OBJECT(record.sink_pad_internal_), "xpos",
                         record.fullscreen_xpos_, "ypos",
                         record.fullscreen_ypos_, "zorder", num_sinks_ + 1,
                         NULL);
          } else {
            g_object_set(G_OBJECT(record.sink_pad_internal_), "xpos",
                         record.xpos_, "ypos", record.ypos_, "zorder",
                         record.stream_no_, NULL);
          }
        }
      }
      break;
    }
    default:
      break;
  }
  return GST_PAD_PROBE_OK;
}

GstPadProbeReturn MixerBin::SrcPadCallback(GstPad *pad, GstPadProbeInfo *info) {
  auto event = gst_pad_probe_info_get_event(info);
  switch (GST_EVENT_TYPE(event)) {
    case GST_EVENT_NAVIGATION: {
      const GstStructure *s = gst_event_get_structure(event);
      auto type = gst_structure_get_string(s, "event");
      if (g_str_equal(type, "key-press")) {
        auto key_pressed = gst_structure_get_string(s, "key");
        auto key = g_ascii_strtoll(key_pressed, NULL, 0);
        if (key > 0 && key <= MAX_NUM_INPUTS) {
          FullScreen(key - 1);
        } else {
          TiledView();
        }
      }
      break;
    }
    default:
      break;
  }
  return GST_PAD_PROBE_OK;
}

bool MixerBin::LinkInput(InferencerBin &inputstream) {

  for (size_t i = 0; i < inputstream.GetNumPads(); i++) {
    int col = (num_sinks_ % NUM_TILE_COLS);
    int row = (num_sinks_ / NUM_TILE_COLS);

    int xpos = TILE_WIDTH * col + inputstream.tiled_video_x_;
    int ypos = TILE_HEIGHT * row + inputstream.tiled_video_y_;
    int fullscreen_xpos = inputstream.fullscreen_video_x_;
    int fullscreen_ypos = inputstream.fullscreen_video_y_;

    auto mixer = gst_bin_get_by_name(GST_BIN(bin_), "m");
    auto sink_pad_internal = gst_element_get_request_pad(mixer, "sink_%u");
    g_object_set(G_OBJECT(sink_pad_internal), "xpos", xpos, "ypos", ypos,
                 "zorder", num_sinks_, NULL);

    std::string sink_name = absl::StrCat("my_sink_", num_sinks_);
    auto source_name("inf_bin_src_" + std::to_string(i));
    auto sink_pad = gst_ghost_pad_new(sink_name.c_str(), sink_pad_internal);
    gst_pad_add_probe(
        sink_pad,
        GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,
        reinterpret_cast<GstPadProbeCallback>(+[](
            GstPad *pad, GstPadProbeInfo *info,
            MixerBin *self) -> GstPadProbeReturn {
          return self->SinkPadCallback(pad, info);
        }),
        this, NULL);
    gst_element_add_pad(bin_, sink_pad);

    gst_object_unref(sink_pad_internal);
    gst_object_unref(mixer);

    if (gst_element_link_pads(inputstream.GetBin(), source_name.c_str(), bin_,
                              sink_name.c_str())) {
      InputConnectionRecord record = { num_sinks_, sink_pad, sink_pad_internal,
          xpos, ypos, fullscreen_xpos, fullscreen_ypos, &inputstream, i };
      linked_inputs_[num_sinks_] = record;
      num_sinks_++;
      inputstream.mixer_ = this;
    } else {
      return false;
    }
  }
  return true;
}

MixerBin::MixerBin() {
  auto mixer_bin_src = absl::Substitute(kMixerBinSrc, NUM_TILE_COLS * TILE_WIDTH,
                                        NUM_TILE_ROWS * TILE_HEIGHT);
  ParseBin(mixer_bin_src);
  auto mixer = gst_bin_get_by_name(GST_BIN(bin_), "m");
  auto source_pad = gst_element_get_static_pad(mixer, "src");
  gst_pad_add_probe(
      source_pad,
      GST_PAD_PROBE_TYPE_EVENT_UPSTREAM,
      reinterpret_cast<GstPadProbeCallback>(+[](
          GstPad *pad, GstPadProbeInfo *info,
          MixerBin *self) -> GstPadProbeReturn {
        return self->SrcPadCallback(pad, info);
      }),
      this, NULL);
  gst_object_unref(mixer);
  gst_object_unref(source_pad);
}

MixerBin::~MixerBin() {
}

} /* namespace szd */

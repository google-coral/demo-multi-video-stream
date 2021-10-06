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
 * Inferencer.cpp
 *
 *  Created on: Mar 15, 2021
 *      Author: pnordstrom
 */

#include <memory>
#include <string>
#include <vector>

#include <glib.h>
#include <glib.h>
#include <gst/pbutils/pbutils.h>
#include <gst/video/video.h>
#include <unistd.h>

#include "absl/strings/substitute.h"
#include "InferencerBin.h"
#include "ManufacturingInferencer.h"
#include "PipelinedInferencer.h"
#include "Utility.h"

namespace szd {
constexpr char InferencerBin::kSvgHeaderTemplate[] =
    "<svg viewBox=\"0 0 $0 $1\">";
constexpr char InferencerBin::kSvgFooter[] = "</svg>";
constexpr char InferencerBin::kSvgBox[] =
    "<rect x=\"$0\" y=\"$1\" width=\"$2\" height=\"$3\" "
        "fill-opacity=\"0.0\" "
        "style=\"stroke-width:2;stroke:rgb($4,$5,$6);\"/>";
constexpr char InferencerBin::kSvgText[] =
    "<text x=\"$0\" y=\"$1\" font-size=\"large\" fill=\"$2\">$3</text>";
const std::string InferencerBin::kSvgHeader = absl::Substitute(
    kSvgHeaderTemplate, kSvgWidth, kSvgHeight);

std::string InferencerBin::MakeKeepOutSvg(Utility::Polygon keepout_polygon) {
  std::string polygon_svg;
  polygon_svg = "<polygon points=\"";
  float x, y, lastx = 0.0, lasty = 0.0;
  for (auto line : keepout_polygon.GetLines()) {
    x = line.begin_.x_;
    y = line.begin_.y_;
    polygon_svg = absl::StrCat(polygon_svg, " ", x * InferencerBin::kSvgWidth,
                               ",", y * kSvgHeight);
    lastx = line.end_.x_;
    lasty = line.end_.y_;
  }
  polygon_svg = absl::StrCat(polygon_svg, " ", lastx * kSvgWidth, ",",
                             lasty * kSvgHeight);
  polygon_svg = absl::StrCat(
      polygon_svg, " \" style=\"fill:none;stroke:green;stroke-width:5\" /> ");

  return polygon_svg;
}

std::string InferencerBin::ResultsToSvg(
    const std::vector<DetectionResult> &results) {
  static const int kMaxIntensity = 255;
  std::string boxlist;
  std::string labellist;
  std::string svg;

  for (const auto &result : results) {
    std::string box_str;
    std::string label_str;
    int w, h;
    w = (result.x2 - result.x1) * kSvgWidth;
    h = (result.y2 - result.y1) * kSvgHeight;
    // Checks if this box collided with the keepout.

    if (!keepout_svg_.empty()) {
      // Check for keepout.
      Utility::Box b { result.x1, result.y1, result.x2, result.y2 };
      if (b.CollidedWithPolygon(keepout_polygon_, 1.0)) {
        box_str = absl::Substitute(kSvgBox, result.x1 * kSvgWidth,
                                   result.y1 * kSvgHeight, w, h, kMaxIntensity, 0, 0);  // Red
        label_str = absl::Substitute(
            kSvgText, result.x1 * kSvgWidth, (result.y1 * kSvgHeight) - 5,
            "red", absl::StrCat(result.candidate, ": ", result.score));
      } else {
        box_str = absl::Substitute(kSvgBox, result.x1 * kSvgWidth,
                                   result.y1 * kSvgHeight, w, h, 0, kMaxIntensity, 0);  // Green
        label_str = absl::Substitute(
            kSvgText, result.x1 * kSvgWidth, (result.y1 * kSvgHeight) - 5,
            "lightgreen", absl::StrCat(result.candidate, ": ", result.score));
      }
    } else {
      // Don't check for keepout.
      box_str = absl::Substitute(kSvgBox, result.x1 * kSvgWidth,
                                 result.y1 * kSvgHeight, w, h, 0, kMaxIntensity, 0);  // Green
      label_str = absl::Substitute(
          kSvgText, result.x1 * kSvgWidth, (result.y1 * kSvgHeight) - 5,
          "lightgreen", absl::StrCat(result.candidate, ": ", result.score));
    }
    boxlist = absl::StrCat(boxlist, box_str);
    labellist = absl::StrCat(labellist, label_str);

  }
  svg = absl::StrCat(kSvgHeader, keepout_svg_, boxlist, labellist,
		     kSvgFooter);

  return svg;
}

GstFlowReturn InferencerBin::AppsinkOnNewSample(GstElement *sink) {
  GstSample *sample = NULL;
  GstFlowReturn retval = GST_FLOW_OK;
  std::shared_ptr<void> output_data;
  std::string svg_string;

  switch (auto type = inferencer_->GetInferencerType()) {
    case kPipelined:
      // For pipelined inferencers, the frames are pulled by the DmaAllocator
      // class in InferencerBin.h, and we always have to call Interpretframe
      // otherwise the buffers won't be freed and we end up with a discrepancy
      // between incoming frames and the video
      inferencer_->InterpretFrame(nullptr, 0, tiled_video_width_,
                                  tiled_video_height_, 0, output_data);
      svg_string = ResultsToSvg(
          *std::static_pointer_cast<std::vector<DetectionResult>>(output_data));
      OutputInferenceResult(svg_string);
      break;
    case kSegmentation:
    case kManufacturing:
    case kDetection:
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
          inferencer_->InterpretFrame(info.data, info.size, meta->width,
                                      meta->height, meta->stride[0],
                                      output_data);
          if (type == kSegmentation) {
            auto segmentation_mask =
                std::static_pointer_cast<std::vector<uint8_t>>(output_data);
            OutputSegmentation(segmentation_mask);
          } else {  // (type == kDetection || kManufacturing)
            svg_string = ResultsToSvg(
                *std::static_pointer_cast<std::vector<DetectionResult>>(
                    output_data));
            OutputInferenceResult(svg_string);
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

  void InferencerBin::SetFullScreenCaps(int src_pad) {
    std::string caps = "video/x-raw(memory:GLMemory), width=$0, height=$1";
    caps = absl::Substitute(caps, fullscreen_video_width_,
                            fullscreen_video_height_);

    gst_util_set_object_arg(G_OBJECT(filter_0_), "caps", caps.c_str());
    fullscreen_ = true;
}

void InferencerBin::SetTiledViewCaps(int src_pad) {
  std::string caps = "video/x-raw(memory:GLMemory), width=$0, height=$1";
  caps = absl::Substitute(caps, tiled_video_width_, tiled_video_height_);

  gst_util_set_object_arg(G_OBJECT(filter_0_), "caps", caps.c_str());
  fullscreen_ = false;
}

void InferencerBin::OutputInferenceResult(const std::string output) {
  g_object_set(G_OBJECT(rsvg_overlay_), "data", output.c_str(), NULL);
}

void InferencerBin::OutputSegmentation(
    std::shared_ptr<std::vector<uint8_t>> segmentation_mask) {
  if (!segmentation_mask->empty()) {
    segmentation_mask_ = segmentation_mask;
  } else {
    segmentation_mask_ = nullptr;
  }

}

void InferencerBin::Rewind() {
  gst_element_seek(decoder_, 1.0, GST_FORMAT_TIME, GST_SEEK_FLAG_SEGMENT,
                   GST_SEEK_TYPE_SET, 0, GST_SEEK_TYPE_NONE, 0);
}

GstPadProbeReturn InferencerBin::QueueSinkPadCallback(GstPad *pad,
                                                      GstPadProbeInfo *info) {
  auto event = gst_pad_probe_info_get_event(info);
  if (GST_EVENT_TYPE(event) == GST_EVENT_SEGMENT_DONE) {
    g_idle_add(reinterpret_cast<GSourceFunc>(+[](InferencerBin *self) -> int {
      self->Rewind();
      return 0;
    }),
               this);
  } else if (GST_EVENT_TYPE(event) == GST_EVENT_RECONFIGURE) {
    return GST_PAD_PROBE_DROP;
  }
  return GST_PAD_PROBE_PASS;
}

const gchar *kFragmentShaderSrc =
  "#ifdef GL_ES\n"
  "precision mediump float;\n"
  "#endif\n"
  "varying vec2 v_texcoord;\n"
  "uniform sampler2D tex;\n"
  "void main()\n"
  "{\n"
  "  vec4 s = texture2D(tex, v_texcoord);\n"
  "  float idxf = s.r * 255.0;\n"
  "  int idx = int(sign(idxf) * floor(abs(idxf) + 0.5));\n"
  "  if (idx == $0) {\n"
  "    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);\n" // detection object color set here
  "  }\n"
  "  else {\n"
  "    gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);\n"
  "  }\n"
  "}";

gboolean InferencerBin::OnClientDraw(GstElement *filter, GLuint in_tex,
                                     GLuint width, GLuint height,
                                     gpointer data) {
  static const unsigned int OVERLAY_W = inferencer_->GetInputWidth();
  static const unsigned int OVERLAY_H = inferencer_->GetInputHeight();
  static const unsigned int OVERLAY_PX_BYTES = OVERLAY_W * OVERLAY_H;
  if (segmentation_mask_ && segmentation_mask_->size() == OVERLAY_PX_BYTES) {
    GstGLContext *context = GST_GL_BASE_FILTER (filter)->context;
    const GstGLFuncs *gl = context->gl_vtable;

    // Compile our custom shader on first use and cache the result.
    if (shader_ == nullptr) {
      GError *error = NULL;
      auto fragment_shader_code = absl::Substitute(kFragmentShaderSrc,
                                                   inferencer_->GetDetectionObject());
      GstGLSLStage *frag_stage = gst_glsl_stage_new_with_string (context,
                                                                 GL_FRAGMENT_SHADER,
                                                                 GST_GLSL_VERSION_NONE,
                                                                 (GstGLSLProfile) (GST_GLSL_PROFILE_COMPATIBILITY | GST_GLSL_PROFILE_ES),
                                                                 fragment_shader_code.c_str());
      GstGLSLStage *vert_stage = gst_glsl_stage_new_default_vertex (context);
      if (!frag_stage || !vert_stage) {
        g_printerr ("Failed to create shader stages\n");
        return FALSE;
      }
      shader_ = gst_gl_shader_new (context);
      if (!gst_gl_shader_compile_attach_stage (shader_, vert_stage, &error)) {
        g_printerr ("Failed to compile vertex shader:\n%s\n", error->message);
        g_clear_error (&error);
        return FALSE;
      }
      if (!gst_gl_shader_compile_attach_stage (shader_, frag_stage, &error)) {
        g_printerr ("Failed to compile fragment shader:\n%s\n", error->message);
        g_clear_error (&error);
        return FALSE;
      }
      if (!gst_gl_shader_link (shader_, NULL)) {
        g_printerr ("Failed to link shader\n");
        return FALSE;
      }
    }

    // Draw the incoming frame.
    gl->Disable(GL_DEPTH_TEST);
    gl->Enable(GL_BLEND);
    gl->BlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    gl->ActiveTexture(GL_TEXTURE0);
    gl->BindTexture(GL_TEXTURE_2D, in_tex);
    gst_gl_shader_use(GST_GL_FILTER (filter)->default_shader);
    gst_gl_filter_draw_fullscreen_quad(GST_GL_FILTER(filter));

    // Second overlay texture with inference data
    // Create overlay texture.
    GLuint o_tex;
    gl->GenTextures(1, &o_tex);
    gl->BindTexture(GL_TEXTURE_2D, o_tex);
    gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    gl->TexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    gl->PixelStorei(GL_UNPACK_ALIGNMENT, 1);
    gl->TexImage2D(GL_TEXTURE_2D, 0, GL_R8, OVERLAY_W, OVERLAY_H, 0, GL_RED,
    GL_UNSIGNED_BYTE,
                   segmentation_mask_->data());
    // Draw the overlay texture.
    gl->ActiveTexture(GL_TEXTURE0);
    gl->BindTexture(GL_TEXTURE_2D, o_tex);
    gst_gl_shader_use(shader_);
    gst_gl_filter_draw_fullscreen_quad(GST_GL_FILTER(filter));
    gl->Disable(GL_BLEND);
    // Free the overlay texture.
    gl->DeleteTextures(1, &o_tex);
  }

  return TRUE;
}

void InferencerBin::SetupAllDims(std::string video_file) {
  char dir[FILENAME_MAX];
  auto uri = absl::StrCat("file:///", getcwd(dir,FILENAME_MAX), "/" ,video_file);

  auto discoverer = gst_discoverer_new(GST_SECOND * 10, nullptr);
  auto info = gst_discoverer_discover_uri(discoverer, uri.c_str(), nullptr);
  auto streaminfo = gst_discoverer_info_get_video_streams(info);

  int video_file_width = gst_discoverer_video_info_get_width(
      GST_DISCOVERER_VIDEO_INFO(streaminfo->data));
  int video_file_height = gst_discoverer_video_info_get_height(
      GST_DISCOVERER_VIDEO_INFO(streaminfo->data));
  auto par_denom = gst_discoverer_video_info_get_par_denom(
      GST_DISCOVERER_VIDEO_INFO(streaminfo->data));
  auto par_num = gst_discoverer_video_info_get_par_num(
      GST_DISCOVERER_VIDEO_INFO(streaminfo->data));
  video_file_height = video_file_height * par_denom / par_num;

  GstVideoRectangle s_rect = { 0, 0, video_file_width, video_file_height };
  GstVideoRectangle d_rect = { 0, 0, TILE_WIDTH, TILE_HEIGHT };
  GstVideoRectangle r_rect = { 0 };

  gst_video_sink_center_rect(s_rect, d_rect, &r_rect, TRUE);
  tiled_video_x_ = r_rect.x;
  tiled_video_y_ = r_rect.y;
  tiled_video_width_ = r_rect.w;
  tiled_video_height_ = r_rect.h;

  s_rect = { 0, 0, video_file_width, video_file_height };
  d_rect = { 0, 0, 3 * TILE_WIDTH, 2 * TILE_HEIGHT };
  r_rect = { 0 };
  gst_video_sink_center_rect(s_rect, d_rect, &r_rect, TRUE);
  fullscreen_video_x_ = r_rect.x;
  fullscreen_video_y_ = r_rect.y;
  fullscreen_video_width_ = r_rect.w;
  fullscreen_video_height_ = r_rect.h;
}

void InferencerBin::SetupBin(std::string bin_src, std::string video_file) {
  ParseBin(bin_src);
  auto source = gst_element_factory_make("filesrc", "source");
  g_object_set(G_OBJECT(source), "location", video_file.c_str(), NULL);

  gst_bin_add(GST_BIN(bin_), source);

  decoder_ = gst_bin_get_by_name(GST_BIN(bin_), "decoder");
  gst_element_link(source, decoder_);

  filter_0_ = gst_bin_get_by_name(GST_BIN(bin_), "filter_0");

  // setup pad probe for enabling looping of videos
  auto q = gst_bin_get_by_name(GST_BIN(bin_), "q");
  auto sink_pad_queue = gst_element_get_static_pad(q, "sink");
  gst_pad_add_probe(
      sink_pad_queue,
      static_cast<GstPadProbeType>(GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM
          | GST_PAD_PROBE_TYPE_EVENT_UPSTREAM),
      reinterpret_cast<GstPadProbeCallback>(+[](
          GstPad *pad, GstPadProbeInfo *info,
          InferencerBin *self) -> GstPadProbeReturn {
        return self->QueueSinkPadCallback(pad, info);
      }),
      this, NULL);
  gst_object_unref(sink_pad_queue);
  gst_object_unref(q);

  rsvg_overlay_ = gst_bin_get_by_name(GST_BIN(bin_), "rsvg_0");
  text_overlay_0_ = gst_bin_get_by_name(GST_BIN(bin_), "text_0");
  g_object_set(G_OBJECT(text_overlay_0_), "text",
               inferencer_->GetModelDescription().c_str(), NULL);

}

InferencerBin::InferencerBin(std::shared_ptr<InferencerBase> inferencer,
                             std::string video_file)
    :
    inferencer_(inferencer) {

  SetupAllDims(video_file);
  auto bin_src = absl::Substitute(inferencer_bin_src_, tiled_video_width_,
                                  tiled_video_height_,
                                  inferencer_->GetInputWidth(),
                                  inferencer_->GetInputHeight());
  SetupBin(bin_src, video_file);

  // Setup the appsink
  auto appsink = gst_bin_get_by_name(GST_BIN(bin_), "appsink_0");
  g_object_set(appsink, "emit-signals", true, NULL);
  g_signal_connect(
      appsink,
      "new-sample",
      reinterpret_cast<GCallback>(+[](GstElement *sink,
                                      InferencerBin *self) -> GstFlowReturn {
        return self->AppsinkOnNewSample(sink);
      }),
      this);
  gst_object_unref(appsink);

  auto segmask = gst_bin_get_by_name(GST_BIN(bin_), "segmask");
  // Setup callbacks etc for non-trivial inferencers
  switch (inferencer_->GetInferencerType()) {
    case kPipelined: {
      allocator_.UpdateAppSink(appsink);
      inferencer_->InitializePipelineRunner(
          &allocator_, [this](const std::string output) {
            this->OutputInferenceResult(output);
          });
      break;
    }
    case kManufacturing: {
      keepout_polygon_ = inferencer_->GetKeepOut();
      keepout_svg_ = MakeKeepOutSvg(keepout_polygon_);
      break;
    }
    case kSegmentation: {
      g_signal_connect(
          G_OBJECT(segmask),
          "client-draw",
          reinterpret_cast<GCallback>(+[](GstElement *filter, GLuint in_tex, GLuint width, GLuint height, InferencerBin *self) -> gboolean { return self->OnClientDraw(filter, in_tex, width, height, NULL); }),
          this);
      break;
    default:
      break;
    }
  }

  // Setup the ouput pad to connect to the mixer
  auto source_pad_internal = gst_element_get_static_pad(filter_0_, "src");
  auto source_pad = gst_ghost_pad_new("inf_bin_src_0", source_pad_internal);
  gst_element_add_pad(bin_, source_pad);
  gst_object_unref(source_pad_internal);
  gst_object_unref(segmask);

  num_src_pads_ = 1;

}

InferencerBin::~InferencerBin() {
  gst_object_unref(filter_0_);
  gst_object_unref(decoder_);
  gst_object_unref(rsvg_overlay_);
  gst_object_unref(text_overlay_0_);
}

} /* namespace szd */

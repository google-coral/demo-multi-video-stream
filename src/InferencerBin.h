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
 * Inferencer.h
 *
 *  Created on: Mar 15, 2021
 *      Author: pnordstrom
 */

#ifndef INFERENCERBIN_H_
#define INFERENCERBIN_H_

#include <gst/allocators/gstdmabuf.h>
#include <gst/gl/gstglfilter.h>
#include <gst/gl/gstglfuncs.h>
#include <gst/gst.h>
#include <sys/mman.h>

#include "coral/pipeline/pipelined_model_runner.h"

#include "Bin.h"
#include "DetectionInferencer.h"
#include "InferencerBase.h"
#include "MixerBin.h"
#include "Utility.h"

namespace szd {
  const std::string kInferencerBinSrc =
      "decodebin name=decoder ! queue name=q ! videoconvert ! videoscale ! tee name=t "
          "t. ! queue ! videoconvert ! "
          "rsvgoverlay fit-to-frame=true name=rsvg_0 ! textoverlay name=text_0 ! videoconvert ! video/x-raw,format=RGBA,width=$0,height=$1 ! "
          "glupload ! glfilterapp name=segmask ! capsfilter name=filter_0 caps=video/x-raw(memory:GLMemory),width=$0,height=$1 "
          "t. ! videoconvert ! videoscale ! video/x-raw,width=$2,height=$3,format=RGB ! "
          "queue leaky=downstream max-size-buffers=1 ! appsink name=appsink_0";

class InferencerBin : public Bin {
 public:
  InferencerBin(std::shared_ptr<InferencerBase> inferencer,
                std::string video_file);
  InferencerBin(InferencerBin &&other) = delete;
  InferencerBin& operator=(const InferencerBin &other) = delete;
  InferencerBin& operator=(InferencerBin &&other) = delete;
  virtual ~InferencerBin();

  void Rewind();

 protected:
  // This constructor needed by TwoModelInferencer child class
  InferencerBin(std::shared_ptr<InferencerBase> inferencer)
      :
      inferencer_(inferencer) {
  }
  std::string ResultsToSvg(const std::vector<DetectionResult> &results);
  void OutputInferenceResult(const std::string output);
  void SetupAllDims(std::string video_file);
  void SetupBin(std::string bin_src, std::string video_file);

  std::shared_ptr<InferencerBase> inferencer_;
  GstElement *filter_0_;
  GstElement *rsvg_overlay_;
  GstElement *text_overlay_0_;
  int tiled_video_x_;
  int tiled_video_y_;
  int tiled_video_width_;
  int tiled_video_height_;
  int fullscreen_video_x_;
  int fullscreen_video_y_;
  int fullscreen_video_width_;
  int fullscreen_video_height_;
  bool fullscreen_ = false;
  size_t num_src_pads_ = 0;
  class MixerBin *mixer_;

 private:
  const std::string inferencer_bin_src_ = kInferencerBinSrc;
  static const int kSvgWidth = TILE_WIDTH;
  static const int kSvgHeight = TILE_HEIGHT;
  static const char kSvgHeaderTemplate[];
  static const char kSvgFooter[];
  static const char kSvgBox[];
  static const char kSvgText[];
  static const std::string kSvgHeader;

  // DmaBuffer and DmaBufferAllocator are helper classes for pipelined inferencer integration
  class DmaAllocator;
  class DmaBuffer : public coral::Buffer {
   public:
    DmaBuffer(GstSample *sample, size_t requested_bytes)
        :
        sample_(CHECK_NOTNULL(sample)),
        requested_bytes_(requested_bytes) {
    }

    // For cases where we can't use DMABuf (like most x64 systems), return the
    // sample. This allows the pipeline runner to see there is no file descriptor
    // and instead rely on inefficient CPU mapping. Ideally this can optimized in
    // the future.
    void* ptr() override {
      GstMapInfo info;
      GstBuffer *buffer = CHECK_NOTNULL(gst_sample_get_buffer(sample_));
      CHECK(gst_buffer_map(buffer, &info, GST_MAP_READ));
      data_ = reinterpret_cast<void*>(info.data);
      return data_;
    }

    void* MapToHost() override {
      if (!handle_) {
        handle_ = mmap(nullptr, requested_bytes_, PROT_READ, MAP_PRIVATE, fd(),
        /*offset=*/0);
        if (handle_ == MAP_FAILED) {
          handle_ = nullptr;
        }
      }
      return handle_;
    }

    bool UnmapFromHost() override {
      if (munmap(handle_, requested_bytes_) != 0) {
        return false;
      }
      return true;
    }

    int fd() {
      if (fd_ == -1) {
        GstBuffer *buf = CHECK_NOTNULL(gst_sample_get_buffer(sample_));
        GstMemory *mem = gst_buffer_peek_memory(buf, 0);
        if (gst_is_dmabuf_memory(mem)) {
          fd_ = gst_dmabuf_memory_get_fd(mem);
        }
      }
      return fd_;
    }

   private:
    friend class DmaAllocator;
    GstSample *sample_ = nullptr;
    size_t requested_bytes_ = 0;

    // DMA Buffer variables
    int fd_ = -1;
    void *handle_ = nullptr;

    // Legacy CPU variables
    void *data_ = nullptr;

  };

  class DmaAllocator : public coral::Allocator {
   public:
    DmaAllocator() = default;
    DmaAllocator(GstElement *sink)
        :
        sink_(CHECK_NOTNULL(sink)) {
    }

    coral::Buffer* Alloc(size_t size_bytes) override {
      GstSample *sample;
      g_signal_emit_by_name(sink_, "pull-sample", &sample);
      return new DmaBuffer(sample, size_bytes);
    }

    void Free(coral::Buffer *buffer) override {
      auto *sample = static_cast<DmaBuffer*>(buffer)->sample_;
      if (sample) {
        gst_sample_unref(sample);
      }

      delete buffer;
    }

    void UpdateAppSink(GstElement *sink) {
      CHECK_NOTNULL(sink);
      sink_ = sink;
    }

   private:
    GstElement *sink_ = nullptr;
  };

  virtual GstFlowReturn AppsinkOnNewSample(GstElement *sink);
  GstPadProbeReturn QueueSinkPadCallback(GstPad *pad, GstPadProbeInfo *info);
  void OutputSegmentation(
      const std::shared_ptr<std::vector<uint8_t>> segmentation_mask);
  virtual void SetFullScreenCaps(int src_pad);
  virtual void SetTiledViewCaps(int src_pad);
  std::shared_ptr<std::vector<uint8_t>> segmentation_mask_ = nullptr;
  gboolean OnClientDraw(GstElement *filter, GLuint in_tex, GLuint width,
                        GLuint height, gpointer data);
  std::string MakeKeepOutSvg(Utility::Polygon keepout_polygon);
  size_t GetNumPads() {
    return num_src_pads_;
  }

  GstElement *decoder_;
  DmaAllocator allocator_;
  Utility::Polygon keepout_polygon_;
  std::string keepout_svg_ = "";
  GstGLShader *shader_ = nullptr;
  friend class MixerBin;
};

} /* namespace szd */

#endif /* INFERENCERBIN_H_ */

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
 * DmaBuffer.h
 *
 *  Created on: Sep 30, 2021
 *      Author: pnordstrom
 */

#ifndef DMABUFFER_H_
#define DMABUFFER_H_

#include <gst/allocators/gstdmabuf.h>
#include <gst/gst.h>
#include <sys/mman.h>

#include "coral/pipeline/pipelined_model_runner.h"

namespace szd {
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

} /* namespace szd */

#endif /* INFERENCERBIN_H_ */

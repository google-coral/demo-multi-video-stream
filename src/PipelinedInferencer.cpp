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
 * MultipleTPUInferencer.cpp
 *
 *  Created on: Apr 16, 2021
 *      Author: pnordstrom
 */
#include "absl/strings/str_format.h"
#include "absl/strings/substitute.h"
#include "coral/tflite_utils.h"

#include "PipelinedInferencer.h"

namespace szd {

void PipelinedInferencer::InterpretFrame(const uint8_t *pixels,
                                         size_t pixel_length, size_t width,
                                         size_t height, size_t stride,
                                         std::shared_ptr<void> &return_data) {
  coral::PipelineTensor input_buffer;
  std::string boxlist;
  std::string labellist;
  std::string svg;

  const TfLiteTensor *input_tensor = interpreter_->input_tensor(0);

  auto alloc = runner_->GetInputTensorAllocator();
  input_buffer.buffer = alloc->Alloc(input_tensor->bytes);

  input_buffer.type = input_tensor->type;
  input_buffer.bytes = input_tensor->bytes;
  input_buffer.name = input_tensor->name;

  if (running_) {
    CHECK(runner_->Push( { input_buffer }).ok());
    mutex_.Lock();
    frames_in_tpu_queue++;
    while (frames_in_tpu_queue >= kMaxQueueSize) {
      cond_.Wait(&mutex_);
    }

    auto results = std::make_shared<std::vector<DetectionResult>>(results_);
    mutex_.Unlock();

    return_data = results;
  }
  return;
}

void PipelinedInferencer::ConsumeRunner() {
  std::vector<coral::PipelineTensor> output_tensors;
  while (runner_->Pop(&output_tensors).ok() && running_) {
    mutex_.Lock();
    frames_in_tpu_queue--;
    cond_.SignalAll();
    std::vector<std::vector<float>> output_data;
    const size_t num_outputs = output_tensors.size();
    CHECK_EQ(num_outputs, 4);
    output_data.resize(num_outputs);

    for (size_t i = 0; i < num_outputs; ++i) {
      const auto *out_tensor = reinterpret_cast<float*>(output_tensors[i].buffer
          ->ptr());
      CHECK_NOTNULL(out_tensor);
      if (output_tensors[i].type == kTfLiteFloat32) {  // detection model out is Float32
        const size_t num_values = output_tensors[i].bytes / sizeof(float);

        output_data[i].resize(num_values);
        for (size_t j = 0; j < num_values; ++j) {
          output_data[i][j] = out_tensor[j];
        }
      }
    }
    results_ = ParseOutputs(output_data);
    mutex_.Unlock();

    for (const auto& tensor : output_tensors) {
      runner_->GetOutputTensorAllocator()->Free(tensor.buffer);
    }
    output_tensors.clear();
  }
}

void PipelinedInferencer::InitializePipelineRunner(
    coral::Allocator *allocator,
    std::function<void(const std::string)> output_cb) {

  output_cb_ = output_cb;
  std::vector<tflite::Interpreter*> runner_interpreters(num_tpus_);

  for (size_t i = 0; i < num_tpus_; ++i) {
    runner_interpreters[i] = segment_interpreters_[i].get();
  }

  runner_ = std::make_unique<coral::PipelinedModelRunner>(runner_interpreters,
                                                          allocator);
  CHECK_NOTNULL(runner_);

  consumer_thread_ = StartConsume();

}

PipelinedInferencer::PipelinedInferencer(const std::string &model_path_base,
                                         const std::string &label_path,
                                         const std::string &detection_object,
                                         const float threshold,
                                         const int num_tpus)
    :
    DetectionInferencer(threshold, detection_object, InferencerBase(num_tpus)) {

  std::vector<std::string> model_path_segments(num_tpus_);
  runner_ = nullptr;

  CHECK_GE(tpu_contexts_.size(), num_tpus_);

  for (size_t i = 0; i < num_tpus_; ++i) {
    CHECK_NOTNULL(tpu_contexts_[i]);
    if (tpu_contexts_[i]->GetDeviceEnumRecord().type
        == edgetpu::DeviceType::kApexPci) {
    }
  }

  for (size_t i = 0; i < num_tpus_; ++i) {
    model_path_segments[i] = absl::Substitute(
        "$0_segment_$1_of_$2_edgetpu.tflite", model_path_base, i, num_tpus_);
  }

  segment_interpreters_.resize(num_tpus_);

  for (size_t i = 0; i < num_tpus_; ++i) {
    models_.push_back(
        CHECK_NOTNULL(
            tflite::FlatBufferModel::BuildFromFile(
                model_path_segments[i].c_str())));
    segment_interpreters_[i] = InitializeInterpreter(models_[i].get(),
                                                     tpu_contexts_[i].get(),
                                                     &error_reporter_);
  }

  Initialize(model_path_segments[0], label_path, detection_object);
  running_ = true;
}

PipelinedInferencer::~PipelinedInferencer() {
  running_ = false;
  runner_->Push({});
  consumer_thread_.join();
}

} /* namespace szd */

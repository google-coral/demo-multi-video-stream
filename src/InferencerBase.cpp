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
 * InferencerBase.cpp
 *
 *  Created on: Apr 7, 2021
 *      Author: pnordstrom
 */
#include <fstream>
#include <regex>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

#include "InferencerBase.h"

namespace szd {

void InferencerBase::InterpretFrame(const uint8_t *pixels, size_t pixel_length,
                                    size_t width, size_t height, size_t stride,
                                    std::shared_ptr<void> &return_data) {
  return_data = nullptr;
  return;
}

std::unique_ptr<tflite::Interpreter> InferencerBase::InitializeInterpreter(
    tflite::FlatBufferModel *model, edgetpu::EdgeTpuContext *context,
    coral::EdgeTpuErrorReporter *error_reporter) {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  tflite::InterpreterBuilder builder(model->GetModel(), resolver, error_reporter);
  std::unique_ptr<tflite::Interpreter> interpreter;
  CHECK_EQ(builder(&interpreter), kTfLiteOk);
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, context);
  CHECK_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  return interpreter;
}

std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord> InferencerBase::all_tpus_;
size_t InferencerBase::next_available_tpu_ = 0;

void InferencerBase::ReadLabels(std::map<int, std::string> &labels,
                                const std::string &label_path,
                                const std::string &detection_object) {
  std::ifstream label_file(label_path);
  if (!label_file.good()) {
    exit(EXIT_FAILURE);
  }

  for (std::string line; getline(label_file, line);) {
    std::istringstream ss(line);
    int id;
    ss >> id;
    // Trim the id and the space from the line to get label.
    line = std::regex_replace(line, std::regex("^[0-9]+ +"), "");
    if (line.compare(detection_object) == 0) {
      detection_object_ = id;
    }
    labels.emplace(id, line);
  }
}

void InferencerBase::Initialize(const std::string &model_path,
                                const std::string &label_path,
                                const std::string &detection_object) {
  model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  CHECK_NOTNULL(model_);
  model_description_ = model_path.substr(model_path.find_last_of("/") + 1);
  model_description_ = absl::StrCat(model_description_, "\non $0 TPU(s)");
  model_description_ = absl::Substitute(model_description_, num_tpus_);

  interpreter_ = InitializeInterpreter(model_.get(), tpu_contexts_[0].get(), &error_reporter_);

  auto dims = interpreter_->input_tensor(0)->dims;
  CHECK_EQ(dims->size, 4);
  input_width_ = dims->data[2];
  input_height_ = dims->data[1];
  input_bytes_ = dims->data[0] * dims->data[1] * dims->data[2] * dims->data[3];

  // sets output tensor shape.
  const auto &out_tensor_indices = interpreter_->outputs();
  output_shape_.resize(out_tensor_indices.size());
  for (size_t i = 0; i < out_tensor_indices.size(); ++i) {
    const auto *tensor = interpreter_->tensor(out_tensor_indices[i]);
    // For detection inferencers the output tensors are only of type float.
    output_shape_[i] = tensor->bytes / sizeof(float);
  }
  ReadLabels(labels_, label_path, detection_object);
}

InferencerBase::InferencerBase(size_t num_tpus)
    :
    InferencerBase() {
  if (num_tpus + next_available_tpu_ > all_tpus_.size()) {
    LOG(ERROR) << "Not enough TPUs found. This demo requires at least 8 TPUs";
    exit(1);
  }

  for (size_t i = next_available_tpu_; i < next_available_tpu_ + num_tpus;
      i++) {
    tpu_contexts_.push_back(
        CHECK_NOTNULL(
            edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
                all_tpus_[i].type, all_tpus_[i].path)));
  }
  next_available_tpu_ += num_tpus;
  num_tpus_ = num_tpus;
}

InferencerBase::InferencerBase() {
  if (all_tpus_.empty()) {
    all_tpus_ = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  }
}

InferencerBase::InferencerBase(const InferencerBase &other) {
  tpu_contexts_ = other.tpu_contexts_;
  num_tpus_ = other.num_tpus_;
}

InferencerBase::~InferencerBase() {
  interpreter_ = nullptr;
}

} /* namespace szd */

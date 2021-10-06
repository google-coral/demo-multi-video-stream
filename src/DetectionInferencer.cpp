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
 * DetectionInferencer.cpp
 *
 *  Created on: Jul 20, 2021
 *      Author: pnordstrom
 */

#include <memory>

#include "absl/strings/substitute.h"

#include "DetectionInferencer.h"

namespace szd {

void DetectionInferencer::InterpretFrame(const uint8_t *pixels,
                                         size_t pixel_length, size_t width,
                                         size_t height, size_t stride,
                                         std::shared_ptr<void> &return_data) {
  auto results = GetDetectionResults(pixels, pixel_length);
  return_data = std::static_pointer_cast<void>(results);
}

std::shared_ptr<std::vector<DetectionResult>> DetectionInferencer::GetDetectionResults(
    const uint8_t *input_data, const int input_size) {
  std::vector<std::vector<float>> output_data;

  uint8_t *input = interpreter_->typed_input_tensor<uint8_t>(0);
  std::memcpy(input, input_data, input_size);

  CHECK_EQ(interpreter_->Invoke(), kTfLiteOk) << error_reporter_.message();

  const auto &output_indices = interpreter_->outputs();
  const size_t num_outputs = output_indices.size();
  output_data.resize(num_outputs);
  for (size_t i = 0; i < num_outputs; ++i) {
    const auto *out_tensor = interpreter_->tensor(output_indices[i]);
    CHECK_NOTNULL(out_tensor);
    if (out_tensor->type == kTfLiteFloat32) {  // detection model out is Float32
      const float *output = interpreter_->typed_output_tensor<float>(i);
      const size_t size_of_output_tensor_i = output_shape_[i];

      output_data[i].resize(size_of_output_tensor_i);
      for (size_t j = 0; j < size_of_output_tensor_i; ++j) {
        output_data[i][j] = output[j];
      }
    }
  }
  return std::make_shared<std::vector<DetectionResult>>(
      ParseOutputs(output_data));
}

std::vector<DetectionResult> DetectionInferencer::ParseOutputs(
    const std::vector<std::vector<float>> &raw_output) {
  std::vector<DetectionResult> results;
  int n = lround(raw_output[3][0]);
  for (int i = 0; i < n; i++) {
    int id = lround(raw_output[1][i]);
    if (id == detection_object_ || detection_object_ < 0) {
      float score = raw_output[2][i];
      if (score > threshold_
          && (detection_object_ < 0 || detection_object_ == id)) {
        DetectionResult result;
        result.candidate = labels_.at(id);
        result.score = score;
        result.y1 = std::max(static_cast<float>(0.0), raw_output[0][4 * i]);
        result.x1 = std::max(static_cast<float>(0.0), raw_output[0][4 * i + 1]);
        result.y2 = std::min(static_cast<float>(1.0), raw_output[0][4 * i + 2]);
        result.x2 = std::min(static_cast<float>(1.0), raw_output[0][4 * i + 3]);
        results.push_back(result);
      }
    }
  }
  return results;
}

DetectionInferencer::DetectionInferencer(const std::string &model_path,
                                         const std::string &label_path,
                                         const float threshold,
                                         const std::string &detection_object)
    :
    InferencerBase(1),
    threshold_(threshold) {
  Initialize(model_path, label_path, detection_object);
}

DetectionInferencer::DetectionInferencer(const std::string &model_path,
                                         const std::string &label_path,
                                         const float threshold,
                                         const std::string &detection_object,
                                         const InferencerBase &other)
    :
    InferencerBase(other),
    threshold_(threshold) {
  Initialize(model_path, label_path, detection_object);
}

DetectionInferencer::DetectionInferencer(const float threshold,
                                         const std::string &detection_object,
                                         const InferencerBase &other)
    :
    InferencerBase(other),
    threshold_(threshold) {
}

DetectionInferencer::~DetectionInferencer() {
}

} /* namespace szd */

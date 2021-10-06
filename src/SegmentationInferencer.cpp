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
 * SegmentationInferencer.cpp
 *
 *  Created on: May 26, 2021
 *      Author: pnordstrom
 */

#include "coral/tflite_utils.h"

#include "SegmentationInferencer.h"

namespace szd {

void SegmentationInferencer::InterpretFrame(
    const uint8_t *pixels, size_t pixel_length, size_t width, size_t height,
    size_t stride, std::shared_ptr<void> &return_data) {
  std::string boxlist;
  std::string labellist;
  std::string svg;

  auto segmask = std::make_shared<std::vector<uint8_t>>();

  GetDetectionResults(pixels, pixel_length, width, height, stride, segmask);
  return_data = segmask;
}

void SegmentationInferencer::GetDetectionResults(
    const uint8_t *input_data, const size_t input_size, const size_t width,
    const size_t height, const size_t stride,
    std::shared_ptr<std::vector<uint8_t>> output_mask) {

  uint8_t *in_tensor = interpreter_->typed_input_tensor<uint8_t>(0);
  for (size_t i = 0; i < height; ++i) {
    std::memcpy(&in_tensor[i * width * 3], &input_data[i * stride], width * 3);
  }

  CHECK_EQ(interpreter_->Invoke(), kTfLiteOk) << error_reporter_.message();

  const TfLiteTensor &out_tensor = *interpreter_->output_tensor(0);

  if (out_tensor.type == kTfLiteInt64) {  // detection model out is Int64
    auto output = coral::TensorData<int64_t>(out_tensor);
    output_mask->resize(height * width);

    for (size_t i = 0; i < height * width; ++i) {
      (*output_mask)[i] = (uint8_t) output[i];
    }
  }
}

SegmentationInferencer::SegmentationInferencer(
    const std::string &model_path, const std::string &label_path,
    const std::string &detection_object, const float threshold)
    :
    InferencerBase(1),
    threshold_(threshold) {
  Initialize(model_path, label_path, detection_object);
}

SegmentationInferencer::~SegmentationInferencer() {
}

} /* namespace szd */

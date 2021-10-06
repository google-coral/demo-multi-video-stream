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
 * ClassificationInferencer.cpp
 *
 *  Created on: Sep 21, 2021
 *      Author: pnordstrom
 */

#include <memory>
#include <string>
#include <vector>

#include "coral/classification/adapter.h"

#include "ClassificationInferencer.h"

namespace szd {

void ClassificationInferencer::InterpretFrame(
    const uint8_t *pixels, size_t pixel_length, size_t width, size_t height,
    size_t stride, std::shared_ptr<void> &return_data) {
  auto result = GetClassificationResults(pixels, pixel_length);
  return_data = std::static_pointer_cast<void>(result);
}

std::shared_ptr<std::vector<ClassificationResult>> ClassificationInferencer::GetClassificationResults(
    const uint8_t *input_data, const int input_size) {
  auto output_data = std::make_shared<std::vector<ClassificationResult>>();

  uint8_t *input = interpreter_->typed_input_tensor<uint8_t>(0);
  std::memcpy(input, input_data, input_size);

  CHECK_EQ(interpreter_->Invoke(), kTfLiteOk) << error_reporter_.message();

  auto results = coral::GetClassificationResults(*interpreter_, threshold_, 1);

  if (!results.empty())
    output_data->push_back( { labels_.at(results[0].id), results[0].score });

  return output_data;

}

ClassificationInferencer::ClassificationInferencer(
    const std::string &model_path, const std::string &label_path,
    const float threshold)
    :
    InferencerBase(1),
    threshold_(threshold) {
  Initialize(model_path, label_path, "");
}

ClassificationInferencer::ClassificationInferencer(
    const std::string &model_path, const std::string &label_path,
    const float threshold, const InferencerBase &other)
    :
    InferencerBase(other),
    threshold_(threshold) {
  Initialize(model_path, label_path, "");
}

ClassificationInferencer::~ClassificationInferencer() {
}

} /* namespace szd */

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
 * ClassificationInferencer.h
 *
 *  Created on: Sep 21, 2021
 *      Author: pnordstrom
 */

#ifndef SRC_CLASSIFICATIONINFERENCER_H_
#define SRC_CLASSIFICATIONINFERENCER_H_

#include "InferencerBase.h"

namespace szd {
struct ClassificationResult {
  std::string candidate;
  float score;
};

class ClassificationInferencer : public szd::InferencerBase {
 public:
  ClassificationInferencer(const std::string &model_path,
                           const std::string &label_path,
                           const float threshold);
  ClassificationInferencer(const std::string &model_path,
                           const std::string &label_path, const float threshold,
                           const InferencerBase &other);
  ClassificationInferencer(const ClassificationInferencer &other) = delete;
  ClassificationInferencer(ClassificationInferencer &&other) = delete;
  ClassificationInferencer& operator=(const ClassificationInferencer &other) = delete;
  ClassificationInferencer& operator=(ClassificationInferencer &&other) = delete;
  virtual ~ClassificationInferencer();

  std::shared_ptr<std::vector<ClassificationResult>> GetClassificationResults(
      const uint8_t *input_data, const int input_size);
  void InterpretFrame(const uint8_t *pixels, size_t pixel_length, size_t width,
                      size_t height, size_t stride,
                      std::shared_ptr<void> &return_data) override;
  virtual InferencerType GetInferencerType() override {
    return kClassification;
  }

 private:
  const float threshold_;
};

} /* namespace szd */

#endif /* SRC_CLASSIFICATIONINFERENCER_H_ */

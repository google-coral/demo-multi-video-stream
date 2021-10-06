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
 * DetectionInferencer.h
 *
 *  Created on: Jul 20, 2021
 *      Author: pnordstrom
 */

#ifndef DETECTIONINFERENCER_H_
#define DETECTIONINFERENCER_H_

#include "absl/strings/substitute.h"

#include "DetectionInferencer.h"
#include "InferencerBase.h"

namespace szd {
struct DetectionResult {
  std::string candidate;
  float score, x1, y1, x2, y2;
};

class DetectionInferencer : public InferencerBase {
 public:
  DetectionInferencer(const std::string &model_path,
                      const std::string &label_path, const float threshold,
                      const std::string &detection_object);
  DetectionInferencer(const std::string &model_path,
                      const std::string &label_path, const float threshold,
                      const std::string &detection_object,
                      const InferencerBase &other);
  virtual ~DetectionInferencer();
  DetectionInferencer() = delete;
  DetectionInferencer(const DetectionInferencer &other) = delete;
  DetectionInferencer(DetectionInferencer &&other) = delete;
  DetectionInferencer& operator=(const DetectionInferencer &other) = delete;
  DetectionInferencer& operator=(DetectionInferencer &&other) = delete;

  void InterpretFrame(const uint8_t *pixels, size_t pixel_length, size_t width,
                      size_t height, size_t stride,
                      std::shared_ptr<void> &return_data) override;
  virtual InferencerType GetInferencerType() override {
    return kDetection;
  }

 protected:
  // This constructor is needed by the PipelinedInferencer so is declared protected
  DetectionInferencer(const float threshold,
                      const std::string &detection_object,
                      const InferencerBase &other);

  std::vector<DetectionResult> ParseOutputs(
      const std::vector<std::vector<float>> &raw_output);

 private:
  const float threshold_;
  std::shared_ptr<std::vector<DetectionResult>> GetDetectionResults(
      const uint8_t *input_data, const int input_size);

};

} /* namespace szd */

#endif /* DETECTIONINFERENCER_H_ */

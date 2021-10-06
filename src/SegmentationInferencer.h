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
 * SegmentationInferencer.h
 *
 *  Created on: May 26, 2021
 *      Author: pnordstrom
 */

#ifndef SEGMENTATIONINFERENCER_H_
#define SEGMENTATIONINFERENCER_H_

#include "InferencerBase.h"

namespace szd {

class SegmentationInferencer : public InferencerBase {
 public:
  SegmentationInferencer(const std::string &model_path,
                         const std::string &label_path,
                         const std::string &detection_object,
                         const float threshold);
  SegmentationInferencer() = delete;
  SegmentationInferencer(const SegmentationInferencer &other) = delete;
  SegmentationInferencer(SegmentationInferencer &&other) = delete;
  SegmentationInferencer& operator=(const SegmentationInferencer &other) = delete;
  SegmentationInferencer& operator=(SegmentationInferencer &&other) = delete;
  virtual ~SegmentationInferencer();

  void InterpretFrame(const uint8_t *pixels, size_t pixel_length, size_t width,
                      size_t height, size_t stride,
                      std::shared_ptr<void> &return_data) override;
  InferencerType GetInferencerType() override {
    return kSegmentation;
  }

 private:
  void GetDetectionResults(const uint8_t *input_data, const size_t input_size,
                           const size_t width, const size_t height,
                           const size_t stride,
                           std::shared_ptr<std::vector<uint8_t>> mask_data);

  const float threshold_;
};

} /* namespace szd */

#endif /* SEGMENTATIONINFERENCER_H_ */

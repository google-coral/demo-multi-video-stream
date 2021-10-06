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
 * MultipleTPUInferencer.h
 *
 *  Created on: Apr 16, 2021
 *      Author: pnordstrom
 */

#ifndef PIPELINEDINFERENCER_H_
#define PIPELINEDINFERENCER_H_

#include "coral/pipeline/pipelined_model_runner.h"

#include "DetectionInferencer.h"
#include "InferencerBase.h"

namespace szd {

class PipelinedInferencer : public DetectionInferencer {
 public:
  PipelinedInferencer() = delete;
  PipelinedInferencer(const PipelinedInferencer &other) = delete;
  PipelinedInferencer(PipelinedInferencer &&other) = delete;
  PipelinedInferencer& operator=(const PipelinedInferencer &other) = delete;
  PipelinedInferencer& operator=(PipelinedInferencer &&other) = delete;
  virtual ~PipelinedInferencer();

  void InterpretFrame(const uint8_t *pixels, size_t pixel_length, size_t width,
                      size_t height, size_t stride,
                      std::shared_ptr<void> &return_data) override;
  InferencerType GetInferencerType() override {
    return kPipelined;
  }
  void InitializePipelineRunner(
      coral::Allocator *allocator,
      std::function<void(const std::string)> output_cb) override;

  PipelinedInferencer(const std::string &model_path_base,
                      const std::string &label_path,
                      const std::string &detection_object,
                      const float threshold, const int num_tpus);

 private:
  static const int kMaxQueueSize = 4;

  void ConsumeRunner();
  std::thread StartConsume() {
    return std::thread([this] {
      ConsumeRunner();
    });
  }

  std::unique_ptr<coral::PipelinedModelRunner> runner_;
  std::vector<std::unique_ptr<tflite::FlatBufferModel>> models_;
  std::vector<std::unique_ptr<tflite::Interpreter>> segment_interpreters_;
  std::vector<DetectionResult> results_;
  absl::Mutex mutex_;
  absl::CondVar cond_;
  int frames_in_tpu_queue = 0;
  std::function<void(const std::string)> output_cb_;
  std::thread consumer_thread_;
  bool running_ = false;
};

}
;
/* namespace szd */

#endif /* PIPELINEDINFERENCER_H_ */

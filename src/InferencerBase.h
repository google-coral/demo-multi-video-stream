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
 * InferencerBase.h
 *
 *  Created on: Apr 7, 2021
 *      Author: pnordstrom
 */
#include <memory>
#include <string>

#include "coral/error_reporter.h"
#include "coral/pipeline/pipelined_model_runner.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tflite/public/edgetpu.h"

#include "Utility.h"

#ifndef INFERENCERBASE_H_
#define INFERENCERBASE_H_

namespace szd {
typedef enum InferencerType {
  kNone,
  kDetection,
  kSegmentation,
  kManufacturing,
  kClassification,
  kPipelined,
} InferencerType;

class InferencerBase {
 public:
  InferencerBase();
  InferencerBase(size_t num_tpus);
  InferencerBase(const InferencerBase &other);
  InferencerBase(InferencerBase &&other) = delete;
  InferencerBase& operator=(const InferencerBase &other) = delete;
  InferencerBase& operator=(InferencerBase &&other) = delete;
  virtual ~InferencerBase();

  virtual void InterpretFrame(const uint8_t *pixels, size_t pixel_length,
                              size_t width, size_t height, size_t stride,
                              std::shared_ptr<void> &return_data);
  virtual InferencerType GetInferencerType() {
    return kNone;
  }
  virtual Utility::Polygon GetKeepOut() {
    return {};
  }
  virtual void InitializePipelineRunner(
      coral::Allocator *allocator,
      std::function<void(const std::string)> output_cb) {
  }
  size_t GetInputWidth() {
    return input_width_;
  }
  size_t GetInputHeight() {
    return input_height_;
  }
  std::string GetModelDescription() {
    return model_description_;
  }
  int GetDetectionObject() {
    return detection_object_;
  }

 protected:
  static std::unique_ptr<tflite::Interpreter> InitializeInterpreter(
      tflite::FlatBufferModel *model, edgetpu::EdgeTpuContext *context, coral::EdgeTpuErrorReporter *error_reporter);
  void Initialize(const std::string &model_path, const std::string &label_path,
                  const std::string &detection_object);
  std::map<int, std::string> labels_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  coral::EdgeTpuErrorReporter error_reporter_;
  std::vector<size_t> output_shape_;
  std::vector<std::shared_ptr<edgetpu::EdgeTpuContext>> tpu_contexts_;
  size_t num_tpus_ = 0;
  int detection_object_ = -1;

 private:
  void ReadLabels(std::map<int, std::string> &labels,
                  const std::string &label_path,
                  const std::string &detection_object);
  static std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord> all_tpus_;
  static size_t next_available_tpu_;
  std::string model_description_ = "No inferencing";
  std::unique_ptr<tflite::FlatBufferModel> model_;
  size_t input_width_ = 1;
  size_t input_height_ = 1;
  size_t input_bytes_ = 1;
};

} /* namespace szd */

#endif /* INFERENCERBASE_H_ */

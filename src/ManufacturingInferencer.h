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
 * ManufacturingInferencer.h
 *
 *  Created on: Apr 9, 2021
 *      Author: pnordstrom
 */

#ifndef MANUFACTURINGINFERENCER_H_
#define MANUFACTURINGINFERENCER_H_

#include "DetectionInferencer.h"
#include "Utility.h"

namespace szd {

class ManufacturingInferencer : public DetectionInferencer {
 public:
  ManufacturingInferencer(const std::string &model_path,
                          const std::string &label_path, const float threshold,
                          const std::string &keepout_path);
  ManufacturingInferencer() = delete;
  ManufacturingInferencer(const ManufacturingInferencer &other) = delete;
  ManufacturingInferencer(ManufacturingInferencer &&other) = delete;
  ManufacturingInferencer& operator=(const ManufacturingInferencer &other) = delete;
  ManufacturingInferencer& operator=(ManufacturingInferencer &&other) = delete;
  virtual ~ManufacturingInferencer();

  InferencerType GetInferencerType() override {
    return kManufacturing;
  }
  Utility::Polygon GetKeepOut() {
    return keepout_polygon_;
  }

 private:
  static Utility::Polygon ParseKeepoutPolygon(const std::string &file_path);
  Utility::Polygon keepout_polygon_;
};

} /* namespace szd */

#endif /* MANUFACTURINGINFERENCER_H_ */

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
 * ManufacturingInferencer.cpp
 *
 *  Created on: Apr 9, 2021
 *      Author: pnordstrom
 */
#include <fstream>
#include <map>
#include <memory>
#include <regex>
#include <string>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"

#include "ManufacturingInferencer.h"

namespace szd {

Utility::Polygon ManufacturingInferencer::ParseKeepoutPolygon(
    const std::string &file_path) {
  std::ifstream f { file_path };
  std::vector<Utility::Point> points;
  if (f.is_open()) {
    // Ignores csv header.
    f.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    for (std::string line; std::getline(f, line);) {
      float x, y;
      std::vector<std::string> p = absl::StrSplit(line, ',');
      CHECK(absl::SimpleAtof(p[0], &x));
      CHECK(absl::SimpleAtof(p[1], &y));
      points.emplace_back(x, y);
    }
    Utility::Polygon keepout_polygon(points);
    return keepout_polygon;
  }
  return {};
}

ManufacturingInferencer::ManufacturingInferencer(
    const std::string &model_path, const std::string &label_path,
    const float threshold, const std::string &keepout_path)
    :
    DetectionInferencer(model_path, label_path, threshold, "person") {

  keepout_polygon_ = ParseKeepoutPolygon(keepout_path);
}

ManufacturingInferencer::~ManufacturingInferencer() {
}

} /* namespace szd */

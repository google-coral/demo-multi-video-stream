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
 * Bin.cpp
 *
 *  Created on: Mar 18, 2021
 *      Author: pnordstrom
 */
#include <glib.h>
#include <gst/gst.h>
#include <string>

#include "Bin.h"

namespace szd {

void Bin::ParseBin(std::string &src, bool link) {
  GError *error = nullptr;
  bin_ = gst_parse_bin_from_description(src.c_str(), link, &error);
}

GstElement* Bin::GetBin() {
  return bin_;
}

Bin::Bin() {
}

Bin::~Bin() {
}

} /* namespace szd */


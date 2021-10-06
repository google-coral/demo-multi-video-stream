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
 * Bin.h
 *
 *  Created on: Mar 18, 2021
 *      Author: pnordstrom
 */

#ifndef BIN_H_
#define BIN_H_

#define TILE_WIDTH (640)
#define TILE_HEIGHT (480)

namespace szd {

class Bin {
 public:
  Bin();
  Bin(const Bin &other) = delete;
  Bin(Bin &&other) = delete;
  Bin& operator=(const Bin &other) = delete;
  Bin& operator=(Bin &&other) = delete;
  virtual ~Bin();

  GstElement* GetBin();

 protected:
  void ParseBin(std::string &src, bool link = false);
  GstElement *bin_;
};

} /* namespace szd */

#endif /* BIN_H_ */

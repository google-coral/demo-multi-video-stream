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
 * MultiVideoStreamsDemo.cpp
 *
 *  Created on: Mar 12, 2021
 *      Author: pnordstrom
 */

#include "Pipeline.h"

namespace szd {
class App {
 public:
  App() = delete;
  App(int argc, char **argv);
  virtual ~App();

  void Run();

 private:
  std::unique_ptr<Pipeline> pipeline_ = nullptr;
};

void App::Run() {
  if (pipeline_) {
    pipeline_->Run();
  }
}

App::App(int argc, char **argv) {
  pipeline_ = std::make_unique<Pipeline>(argc, argv);
}

App::~App() {
}

}  /* namespace szd */

int main(int argc, char **argv) {
  szd::App app(argc, argv);
  app.Run();
  return 0;
}


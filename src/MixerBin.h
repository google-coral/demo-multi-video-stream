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
 * MixerBin.h
 *
 *  Created on: Mar 18, 2021
 *      Author: pnordstrom
 */

#ifndef MIXERBIN_H_
#define MIXERBIN_H_

#include "Bin.h"
#include "InferencerBin.h"

#define MAX_NUM_INPUTS (6)
#define NUM_TILE_COLS (3)
#define NUM_TILE_ROWS (2)

namespace szd {

  const std::string kMixerBinSrc =
      "glvideomixer name=m background=black ! video/x-raw,width=$0,height=$1 ! videoconvert ! glimagesink";

class MixerBin : public Bin {
 public:
  MixerBin();
  MixerBin(const MixerBin &other) = delete;
  MixerBin(MixerBin &&other) = delete;
  MixerBin& operator=(const MixerBin &other) = delete;
  MixerBin& operator=(MixerBin &&other) = delete;
  virtual ~MixerBin();

  bool LinkInput(class InferencerBin &inputstream);
  friend class InferencerBin;
  friend class TwoModelInferencerBin;

 private:
  struct InputConnectionRecord {
    int stream_no_;
    GstPad *sink_pad_external_;
    GstPad *sink_pad_internal_;
    int xpos_;
    int ypos_;
    int fullscreen_xpos_;
    int fullscreen_ypos_;
    class InferencerBin *inputstream;
    size_t src_pad_no_;
  };

  bool DoInterpret(InferencerBin *stream);
  GstPadProbeReturn SrcPadCallback(GstPad *pad, GstPadProbeInfo *info);
  GstPadProbeReturn SinkPadCallback(GstPad *pad, GstPadProbeInfo *info);
  void FullScreen(int stream_no);
  void TiledView();

  std::array<InputConnectionRecord, MAX_NUM_INPUTS> linked_inputs_;
  int num_sinks_ = 0;
  int fullscreen_stream_ = -1;
  bool fullscreen_ = false;
};

} /* namespace szd */

#endif /* MIXERBIN_H_ */

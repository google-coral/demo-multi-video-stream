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
 * TwoModelInferencerBin.h
 *
 *  Created on: Sep 2, 2021
 *      Author: pnordstrom
 */

#ifndef SRC_TWOMODELINFERENCERBIN_H_
#define SRC_TWOMODELINFERENCERBIN_H_

#include "InferencerBin.h"

namespace szd {
  const std::string kTwoModelInferencerBinSrc =
      "decodebin name=decoder ! queue name=q ! videoconvert ! tee name=t0 "
          "t0. ! tee name=t1 "
          "t1. ! videoscale ! capsfilter name=filter_0 caps=video/x-raw,width=$0,height=$1 ! queue ! videoconvert ! "
          "rsvgoverlay fit-to-frame=true name=rsvg_0 ! textoverlay name=text_0 "
          "t1. ! videoconvert ! videoscale ! video/x-raw,width=$2,height=$3,format=RGB ! "
          "queue leaky=downstream max-size-buffers=1 ! appsink name=appsink_0 "
          "t0. ! tee name=t2 "
          "t2. ! videocrop name=cropper ! videoscale ! capsfilter name=filter_1 caps=video/x-raw,width=$0,height=$1,pixel-aspect-ratio=1/1 ! queue ! "
          "videoconvert ! textoverlay name=text_1 "
          "t2. ! videoconvert ! videoscale ! video/x-raw,width=$4,height=$5,format=RGB ! "
          "queue leaky=downstream max-size-buffers=1 ! appsink name=appsink_1";

class TwoModelInferencerBin : public szd::InferencerBin {
 public:
  TwoModelInferencerBin(std::shared_ptr<InferencerBase> first_inferencer,
                        std::shared_ptr<InferencerBase> second_inferencer,
                        std::string video_file);
  TwoModelInferencerBin() = delete;
  TwoModelInferencerBin(const TwoModelInferencerBin &other) = delete;
  TwoModelInferencerBin(TwoModelInferencerBin &&other) = delete;
  TwoModelInferencerBin& operator=(const TwoModelInferencerBin &other) = delete;
  TwoModelInferencerBin& operator=(TwoModelInferencerBin &&other) = delete;
  virtual ~TwoModelInferencerBin();

 private:
  const std::string inferencer_bin_src_ = kTwoModelInferencerBinSrc;
  GstFlowReturn AppsinkOnNewSample(GstElement *sink) override;
  void SetFullScreenCaps(int src_pad) override;
  void SetTiledViewCaps(int src_pad) override;
  GstPadProbeReturn CropperSinkPadCallback(GstPad *pad, GstPadProbeInfo *info);
  std::shared_ptr<InferencerBase> second_inferencer_;
  GstElement *filter_1_;
  GstElement *text_overlay_1_;
  GstElement *appsink_0_;
  GstElement *appsink_1_;
  GstElement *cropper_;
  int cropper_input_width_;
  int cropper_input_height_;
  friend class MixerBin;
};
}  /* namespace szd */
#endif /* SRC_TWOMODELINFERENCERBIN_H_ */

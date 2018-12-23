/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

namespace w2l {

struct SpeechStats {
  long long totalInputSz_;
  long long totalTargetSz_;
  long long maxInputSz_;
  long long maxTargetSz_;
  long long numSamples_;

  SpeechStats();
  void reset();
  std::vector<long long> toArray();
};

class SpeechStatMeter {
 public:
  SpeechStatMeter();
  void add(const af::array& input, const af::array& target);
  void add(const SpeechStats& stats);
  std::vector<long long> value();
  void reset();

 private:
  SpeechStats stats_;
};
} // namespace w2l

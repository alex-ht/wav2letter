/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>
#include "FeatureParams.h"

namespace speech {

// Convert the speech signal into frames
template <typename T>
std::vector<T> frameSignal(
    const std::vector<T>& input,
    const FeatureParams& params);

// row major;  matA - m x k , matB - k x n

/*template <typename T>
std::vector<T> mklGemm(
    const std::vector<T>& matA,
    const std::vector<T>& matB,
    int n,
    int k);*/

std::vector<float> mklGemm(
    const std::vector<float>& matA,
    const std::vector<float>& matB,
    int n,
    int k);

std::vector<double> mklGemm(
    const std::vector<double>& matA,
    const std::vector<double>& matB,
    int n,
    int k);
} // namespace speech

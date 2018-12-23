/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "FullConnectionCriterion.h"

#include "CriterionUtils.h"
#include "math.h"

using namespace fl;

namespace w2l {

FullConnectionCriterion::FullConnectionCriterion(
    long long N,
    w2l::CriterionScaleMode scalemode)
    : N_(N), scaleMode_(scalemode) {
  if (N_ <= 0) {
    throw std::invalid_argument(
        "FCC: Size of transition matrix is less than 0.");
  }
  auto transition = constant(0.0, af::dim4(N_, N_));
  params_ = {transition};
}

Variable FullConnectionCriterion::forward(
    const Variable& input,
    const Variable& target) {
  const long long N = input.dims(0);
  const long long T = input.dims(1);
  const long long B = input.dims(2);
  const long long L = target.dims(0);
  if (N != N_) {
    throw std::invalid_argument("FCC: N doesn't match with the letter size.");
  }

  /* Forward */
  auto fwBuf = fwParams(N, T, B, L);
  target.host(fwBuf.targetsRaw.data());
  input.host(fwBuf.inputsRaw.data());
  params_[0].host(fwBuf.transRaw.data());

  auto scaleFn = getCriterionScaleFn(scaleMode_);

#pragma omp parallel for num_threads(B)
  for (long long b = 0; b < B; b++) {
    auto targets = fwBuf.targetsRaw.data() + b * L;
    long long TN = w2l::getTargetSize(targets, L);
    TN = std::min(TN, T);
    if (TN == 0) {
      throw std::invalid_argument("Target size cannot be empty for FCC");
    }
    fwBuf.scale[b] = scaleFn(N, T, TN);
    double* alpha = fwBuf.alpha.data() + b * N * T;
    long long* alphaIndex = fwBuf.alphaIndex.data() + b * N * T;
    float* inputsRaw = fwBuf.inputsRaw.data() + b * N * T;

    for (long long i = 0; i < N; i++) {
      alpha[i] = inputsRaw[i];
    }

    double* alphaCurFrame;
    long long* alphaIndexCurFrame;
    for (long long t = 1; t < T; t++) {
      double* alphaPrevFrame = alpha + (t - 1) * N;
      alphaCurFrame = alpha + t * N;
      alphaIndexCurFrame = alphaIndex + t * N;
      const float* inputs = inputsRaw + t * N;
      for (long long i = 0; i < N; i++) {
        double sum = 0, max = NEG_INFINITY_DBL;
        for (long long j = 0; j < N; j++) {
          double z = fwBuf.transRaw[i * N + j] + alphaPrevFrame[j];
          if (max < z) {
            alphaIndexCurFrame[i] = j;
            max = z;
          }
        }
        for (long long j = 0; j < N; j++) {
          double z = fwBuf.transRaw[i * N + j] + alphaPrevFrame[j];
          sum += std::exp(z - max);
        }

        alphaCurFrame[i] = max + std::log(sum) + inputs[i];
      }
    }

    alphaCurFrame = alpha + (T - 1) * N;
    double sum = 0, max = NEG_INFINITY_DBL;
    for (long i = 0; i < N; i++) {
      if (max < alphaCurFrame[i]) {
        max = alphaCurFrame[i];
      }
    }
    for (long long i = 0; i < N; i++) {
      sum += std::exp(alphaCurFrame[i] - max);
    }
    fwBuf.res[b] = static_cast<float>((std::log(sum) + max) * fwBuf.scale[b]);
  }

  auto result = af::array(B, fwBuf.res.data());

  /* Backward */
  auto gradFunc = [B, N, T, fwBuf](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    auto bwBuf = bwParams(N, T, B);
    gradOutput.host(bwBuf.outputsGrad.data());

#pragma omp parallel for num_threads(B)
    for (long long b = 0; b < B; b++) {
      const float grad = fwBuf.scale[b] * bwBuf.outputsGrad[b];
      float* inputsGrad = bwBuf.inputsGrad.data() + b * N * T;
      double* alphaGrad = bwBuf.alphaGrad.data() + b * N * T;
      const double* alpha = fwBuf.alpha.data() + b * N * T;
      const long long* alphaIndex = fwBuf.alphaIndex.data() + b * N * T;
      double* transGrad = bwBuf.transGrad.data() + b * N * N;

      // bw step 1
      {
        const double* alphaCurFrame = alpha + (T - 1) * N;
        double* alphaGradCurFrame = alphaGrad + (T - 1) * N;
        float* inputsGradCurFrame = inputsGrad + (T - 1) * N;
        double max = NEG_INFINITY_DBL;
        for (long long j = 0; j < N; j++) {
          if (max < alphaCurFrame[j]) {
            max = alphaCurFrame[j];
          }
        }

        double alphaGradSum = 0;
        for (long long j = 0; j < N; j++) {
          alphaGradSum += std::exp(alphaCurFrame[j] - max);
        }
        for (long long j = 0; j < N; j++) {
          alphaGradCurFrame[j] =
              std::exp(alphaCurFrame[j] - max) / alphaGradSum;
          inputsGradCurFrame[j] = alphaGradCurFrame[j] * grad;
        }
      }

      // bw
      for (long long t = T - 2; t >= 0; t--) {
        const double* alphaCurFrame = alpha + t * N;
        double* alphaGradCurFrame = alphaGrad + t * N;
        double* alphaGradPrevFrame = alphaGrad + (t + 1) * N;
        const long long* alphaIndexCurFrame = alphaIndex + (t + 1) * N;
        float* inputsGradCurFrame = inputsGrad + t * N;

        std::vector<double> m(N * N);
        for (long long i = 0; i < N; i++) {
          double max = fwBuf.transRaw[i * N + alphaIndexCurFrame[i]] +
              alphaCurFrame[alphaIndexCurFrame[i]];
          double alphaGradSum = 0;
          for (long long j = 0; j < N; j++) {
            m[i * N + j] =
                std::exp(fwBuf.transRaw[i * N + j] + alphaCurFrame[j] - max);
            alphaGradSum += m[i * N + j];
          }
          for (long long j = 0; j < N; j++) {
            m[i * N + j] = m[i * N + j] / alphaGradSum;
          }
        }

        for (long long i = 0; i < N; i++) {
          for (long long j = 0; j < N; j++) {
            alphaGradCurFrame[i] += m[j * N + i] * alphaGradPrevFrame[j];
            transGrad[j * N + i] += m[j * N + i] * alphaGradPrevFrame[j] * grad;
          }
          inputsGradCurFrame[i] = alphaGradCurFrame[i] * grad;
        }
      }
    }

    for (long long b = 0; b < B; b++) {
      double* transGrad = bwBuf.transGrad.data() + b * N * N;
      for (long long i = 0; i < N * N; i++) {
        bwBuf.transGradRes[i] += static_cast<float>(transGrad[i]);
      }
    }

    inputs[0].addGrad(
        Variable(af::array(N, T, B, bwBuf.inputsGrad.data()), false));
    inputs[1].addGrad(
        Variable(af::array(N, N, bwBuf.transGradRes.data()), false));
  };

  return Variable(result, {input, params_[0]}, gradFunc);
}

std::string FullConnectionCriterion::prettyString() const {
  return "FullConnectionCriterion";
}

} // namespace w2l

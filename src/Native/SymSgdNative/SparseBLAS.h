// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once
#include "../Stdafx.h"

extern "C" float cblas_sdot(const int vecSize, const float* denseVecX, const int incX, const float* denseVecY, const int incY);
extern "C" float cblas_sdoti(const int sparseVecSize, const float* sparseVecValues, const int* sparseVecIndices, float* denseVec);
extern "C" void cblas_saxpy(const int vecSize, const float coef, const float* denseVecX, const int incX, float* denseVecY, const int incY);
extern "C" void cblas_saxpyi(const int sparseVecSize, const float coef, const float* sparseVecValues, const int* sparseVecIndices, float* denseVec);

float SDOT(const int vecSize, const float* denseVecX, const float* denseVecY) {
    return cblas_sdot(vecSize, denseVecX, 1, denseVecY, 1);
}

float SDOTI(const int sparseVecSize, const int* sparseVecIndices, const float* sparseVecValues, float* denseVec) {
    return cblas_sdoti(sparseVecSize, sparseVecValues, sparseVecIndices, denseVec);
}

void SAXPY(const int vecSize, const float* denseVecX, float* denseVecY, float coef) {
    return cblas_saxpy(vecSize, coef, denseVecX, 1, denseVecY, 1);
}

void SAXPYI(const int sparseVecSize, const int* sparseVecIndices, const float* sparseVecValues, float* denseVec, float coef) {
    cblas_saxpyi(sparseVecSize, coef, sparseVecValues, sparseVecIndices, denseVec);
}
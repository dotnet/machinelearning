// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once
#include "../Stdafx.h"

EXPORT_API(int) simpleTest(void * features, int nColumns);

EXPORT_API(void) linearRegressionSingle(void * features, void * label, void * betas, int nRows, int nColumns);
EXPORT_API(int) ridgeRegressionOnlineCompute(void * featuresPtr, void * labelsPtr, int nRows, int nColumns, float l2Reg, void * partialResultPtr, int partialResultSize);
EXPORT_API(void) ridgeRegressionOnlineFinalize(void * featuresPtr, void * labelsPtr, int nAllRows, int nRows, int nColumns, float l2Reg, void * partialResultPtr, int partialResultSize,
    void * betaPtr, void * xtyPtr, void * xtxPtr);
EXPORT_API(void) linearRegressionDouble(void * features, void * label, void * betas, int nRows, int nColumns);

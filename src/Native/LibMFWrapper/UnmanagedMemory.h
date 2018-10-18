// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once
#include "mf.h"

using namespace mf;

#ifdef _MSC_VER
#define EXPORT_API(ret) extern "C" __declspec(dllexport) ret __stdcall
#else
#define EXPORT_API(ret) extern "C" __attribute__((visibility("default"))) ret
#endif

EXPORT_API(void) MFDestroyModel(mf_model *&model);

EXPORT_API(mf_model*) MFTrain(const mf_problem *prob, const mf_parameter *param);

EXPORT_API(mf_model*) MFTrainWithValidation(const mf_problem *tr, const mf_problem *va, const mf_parameter *param);
    
EXPORT_API(float) MFCrossValidation(const mf_problem *prob, int nr_folds, const mf_parameter* param);

EXPORT_API(float) MFPredict(const mf_model *model, int p_idx, int q_idx);

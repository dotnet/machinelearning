// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once
#include "mf.h"
#include "../Stdafx.h"

using namespace mf;

EXPORT_API(void) MFDestroyModel(mf_model *&model);

EXPORT_API(mf_model*) MFTrain(const mf_problem *prob, const mf_parameter *param);

EXPORT_API(mf_model*) MFTrainWithValidation(const mf_problem *tr, const mf_problem *va, const mf_parameter *param);
    
EXPORT_API(float) MFCrossValidation(const mf_problem *prob, int nr_folds, const mf_parameter* param);

EXPORT_API(float) MFPredict(const mf_model *model, int p_idx, int q_idx);

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once
#include "mf.h"
#include "../Stdafx.h"

using namespace mf;

struct mf_parameter_bridge
{
    int32_t fun;
    int32_t k;
    int32_t nr_threads;
    int32_t nr_bins;
    int32_t nr_iters;
    float lambda_p1;
    float lambda_p2;
    float lambda_q1;
    float lambda_q2;
    float eta;
    float alpha;
    float c;
    uint8_t do_nmf;
    uint8_t quiet;
    uint8_t copy_data;
};

struct mf_problem_bridge
{
    int32_t m;
    int32_t n;
    int64_t nnz;
    struct mf_node *R;
};

struct mf_model_bridge
{
    int32_t fun;
    int32_t m;
    int32_t n;
    int32_t k;
    float b;
    float *P;
    float *Q;
};

EXPORT_API(void) MFDestroyModel(mf_model_bridge *&model);

EXPORT_API(mf_model_bridge*) MFTrain(const mf_problem_bridge *prob_bridge, const mf_parameter_bridge *parameter_bridge);

EXPORT_API(mf_model_bridge*) MFTrainWithValidation(const mf_problem_bridge *tr, const mf_problem_bridge *va, const mf_parameter_bridge *parameter_bridge);
    
EXPORT_API(float) MFCrossValidation(const mf_problem_bridge *prob, int32_t nr_folds, const mf_parameter_bridge* parameter_bridge);

EXPORT_API(float) MFPredict(const mf_model_bridge *model, int32_t p_idx, int32_t q_idx);

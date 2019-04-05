// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#include <stdlib.h>

#include "UnmanagedMemory.h"
#include "mf.h"

using namespace mf;

mf_parameter make_param(const mf_parameter_bridge *param_bridge)
{
    mf_parameter param;
    param.fun = param_bridge->fun;
    param.k = param_bridge->k;
    param.nr_threads = param_bridge->nr_threads;
    param.nr_bins = param_bridge->nr_bins;
    param.nr_iters = param_bridge->nr_iters;
    param.lambda_p1 = param_bridge->lambda_p1;
    param.lambda_p2 = param_bridge->lambda_p2;
    param.lambda_q1 = param_bridge->lambda_q1;
    param.lambda_q2 = param_bridge->lambda_q2;
    param.eta = param_bridge->eta;
    param.alpha = param_bridge->alpha;
    param.c = param_bridge->c;
    param.do_nmf = param_bridge->do_nmf != 0 ? true : false;
    param.quiet = param_bridge->quiet != 0 ? true : false;
    param.copy_data = param_bridge->copy_data != 0 ? true : false;
    return param;
}

EXPORT_API(void) MFDestroyModel(mf_model *&model)
{
    return mf_destroy_model(&model);
}

EXPORT_API(mf_model*) MFTrain(const mf_problem *prob, const mf_parameter_bridge *param_bridge)
{
    auto param = make_param(param_bridge);
    return mf_train(prob, param);
}

EXPORT_API(mf_model*) MFTrainWithValidation(const mf_problem *tr, const mf_problem *va, const mf_parameter_bridge *param_bridge)
{
    auto param = make_param(param_bridge);
    return mf_train_with_validation(tr, va, param);
}

EXPORT_API(float) MFCrossValidation(const mf_problem *prob, int nr_folds, const mf_parameter_bridge *param_bridge)
{
    auto param = make_param(param_bridge);
    return mf_cross_validation(prob, nr_folds, param);
}

EXPORT_API(float) MFPredict(const mf_model *model, int p_idx, int q_idx)
{
    return mf_predict(model, p_idx, q_idx);
}

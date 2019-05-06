// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#include <stdlib.h>

#include "UnmanagedMemory.h"
#include "mf.h"

using namespace mf;

inline mf_parameter TranslateToParam(const mf_parameter_bridge *param_bridge)
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

inline mf_problem TranslateToProblem(const mf_problem_bridge *prob_bridge)
{
    mf_problem prob;
    prob.m = prob_bridge->m;
    prob.n = prob_bridge->n;
    prob.nnz = prob_bridge->nnz;
    prob.R = prob_bridge->R;
    return prob;
}

inline void TranslateToModelBridge(const mf_model *model, mf_model_bridge *model_bridge)
{
    model_bridge->fun = model->fun;
    model_bridge->m = model->m;
    model_bridge->n = model->n;
    model_bridge->k = model->k;
    model_bridge->b = model->b;
    model_bridge->P = model->P;
    model_bridge->Q = model->Q;
}

inline void TranslateToModel(const mf_model_bridge *model_bridge, mf_model *model)
{
    model->fun = model_bridge->fun;
    model->m = model_bridge->m;
    model->n = model_bridge->n;
    model->k = model_bridge->k;
    model->b = model_bridge->b;
    model->P = model_bridge->P;
    model->Q = model_bridge->Q;
}

EXPORT_API(void) MFDestroyModel(mf_model_bridge *&model_bridge)
{
    // Transfer the ownership of P and Q back to the original LIBMF class, so that
    // mf_destroy_model can be called.
    auto model = new mf_model;
    model->P = model_bridge->P;
    model->Q = model_bridge->Q;
    mf_destroy_model(&model); // delete model, model->P, amd model->Q.

    // Delete bridge class allocated in MFTrain, MFTrainWithValidation, or MFCrossValidation.
    delete model_bridge;
    model_bridge = nullptr;
}

EXPORT_API(mf_model_bridge*) MFTrain(const mf_problem_bridge *prob_bridge, const mf_parameter_bridge *param_bridge)
{
    // Convert objects created outside LIBMF. Notice that the called LIBMF function doesn't take the ownership of
    // allocated memory in those external objects.
    auto prob = TranslateToProblem(prob_bridge);
    auto param = TranslateToParam(param_bridge);

    // The model contains 3 allocated things --- itself, P, and Q.
    // We will delete itself and transfer the ownership of P and Q to the associated bridge class. The bridge class
    // will then be sent to C#.
    auto model = mf_train(&prob, param);
    auto model_bridge = new mf_model_bridge;
    TranslateToModelBridge(model, model_bridge);
    delete model;
    return model_bridge; // To clean memory up, we need to delete model_bridge, model_bridge->P, and model_bridge->Q.
}

EXPORT_API(mf_model_bridge*) MFTrainWithValidation(const mf_problem_bridge *tr_bridge, const mf_problem_bridge *va_bridge, const mf_parameter_bridge *param_bridge)
{
    // Convert objects created outside LIBMF. Notice that the called LIBMF function doesn't take the ownership of
    // allocated memory in those external objects.
    auto tr = TranslateToProblem(tr_bridge);
    auto va = TranslateToProblem(va_bridge);
    auto param = TranslateToParam(param_bridge);

    // The model contains 3 allocated things --- itself, P, and Q.
    // We will delete itself and transfer the ownership of P and Q to the associated bridge class. The bridge class
    // will then be sent to C#.
    auto model = mf_train_with_validation(&tr, &va, param);
    auto model_bridge = new mf_model_bridge;
    TranslateToModelBridge(model, model_bridge);
    delete model;
    return model_bridge; // To clean memory up, we need to delete model_bridge, model_bridge->P, and model_bridge->Q.
}

EXPORT_API(float) MFCrossValidation(const mf_problem_bridge *prob_bridge, int32_t nr_folds, const mf_parameter_bridge *param_bridge)
{
    auto param = TranslateToParam(param_bridge);
    auto prob = TranslateToProblem(prob_bridge);
    return static_cast<float>(mf_cross_validation(&prob, nr_folds, param));
}

EXPORT_API(float) MFPredict(const mf_model_bridge *model_bridge, int32_t p_idx, int32_t q_idx)
{
    mf_model model;
    TranslateToModel(model_bridge, &model);
    return mf_predict(&model, p_idx, q_idx);
}

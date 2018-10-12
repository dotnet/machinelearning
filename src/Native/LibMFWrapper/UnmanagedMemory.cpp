//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
#include <malloc.h>

#include "UnmanagedMemory.h"
#include "mf.h"

using namespace mf;

EXPORT_API(void) MFDestroyModel(mf_model *&model)
{
    return mf_destroy_model(&model);
}

EXPORT_API(mf_model*) MFTrain(const mf_problem *prob, const mf_parameter *param)
{
    return mf_train(prob, *param);
}

EXPORT_API(mf_model*) MFTrainWithValidation(const mf_problem *tr, const mf_problem *va, const mf_parameter *param)
{
    return mf_train_with_validation(tr, va, *param);
}


EXPORT_API(float) MFCrossValidation(const mf_problem *prob, int nr_folds, const mf_parameter *param)
{
    return mf_cross_validation(prob, nr_folds, *param);
}

EXPORT_API(float) MFPredict(const mf_model *model, int p_idx, int q_idx)
{
    return mf_predict(model, p_idx, q_idx);
}

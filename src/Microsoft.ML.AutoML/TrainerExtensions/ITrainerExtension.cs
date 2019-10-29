// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Trainers;

namespace Microsoft.ML.AutoML
{
    using ITrainerEstimator = ITrainerEstimator<IPredictionTransformer<object>, object>;

    internal interface ITrainerExtension
    {
        IEnumerable<SweepableParam> GetHyperparamSweepRanges();

        ITrainerEstimator CreateInstance(MLContext mlContext,
            IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo, IDataView validationSet = null);

        PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo);
    }
}

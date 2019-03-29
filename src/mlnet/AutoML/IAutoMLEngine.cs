
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Auto;
using Microsoft.ML.CLI.ShellProgressBar;
using Microsoft.ML.Data;

namespace Microsoft.ML.CLI.CodeGenerator
{
    internal interface IAutoMLEngine
    {
        ColumnInferenceResults InferColumns(MLContext context, ColumnInformation columnInformation);

        IEnumerable<RunResult<BinaryClassificationMetrics>> ExploreBinaryClassificationModels(MLContext context, IDataView trainData, IDataView validationData, ColumnInformation columnInformation, BinaryClassificationMetric optimizationMetric, ProgressBar progressBar);

        IEnumerable<RunResult<MultiClassClassifierMetrics>> ExploreMultiClassificationModels(MLContext context, IDataView trainData, IDataView validationData, ColumnInformation columnInformation, MulticlassClassificationMetric optimizationMetric, ProgressBar progressBar);

        IEnumerable<RunResult<RegressionMetrics>> ExploreRegressionModels(MLContext context, IDataView trainData, IDataView validationData, ColumnInformation columnInformation, RegressionMetric optimizationMetric, ProgressBar progressBar);

    }
}

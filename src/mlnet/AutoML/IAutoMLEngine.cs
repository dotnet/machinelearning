// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.AutoML;
using Microsoft.ML.CLI.ShellProgressBar;
using Microsoft.ML.CLI.Utilities;
using Microsoft.ML.Data;

namespace Microsoft.ML.CLI.CodeGenerator
{
    internal interface IAutoMLEngine
    {
        ColumnInferenceResults InferColumns(MLContext context, ColumnInformation columnInformation);

        void ExploreBinaryClassificationModels(MLContext context, IDataView trainData, IDataView validationData, ColumnInformation columnInformation, BinaryClassificationMetric optimizationMetric, ProgressHandlers.BinaryClassificationHandler handler, ProgressBar progressBar = null);

        void ExploreMultiClassificationModels(MLContext context, IDataView trainData, IDataView validationData, ColumnInformation columnInformation, MulticlassClassificationMetric optimizationMetric, ProgressHandlers.MulticlassClassificationHandler handler, ProgressBar progressBar = null);

        void ExploreRegressionModels(MLContext context, IDataView trainData, IDataView validationData, ColumnInformation columnInformation, RegressionMetric optimizationMetric, ProgressHandlers.RegressionHandler handler, ProgressBar progressBar = null);

    }
}

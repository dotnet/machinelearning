
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML.Auto;

namespace Microsoft.ML.CLI.CodeGenerator
{
    internal interface IAutoMLEngine
    {
        ColumnInferenceResults InferColumns(MLContext context, ColumnInformation columnInformation);

        (Pipeline, ITransformer) ExploreModels(MLContext context, IDataView trainData, IDataView validationData, ColumnInformation columnInformation);

    }
}

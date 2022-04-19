// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace Microsoft.ML.TorchSharp
{
    /// <summary>
    /// Collection of extension methods for <see cref="T:Microsoft.ML.MulticlassClassificationCatalog.MulticlassClassificationTrainers" /> to create instances of TorchSharp trainer components.
    /// </summary>
    /// <remarks>
    /// This requires additional nuget dependencies to link against TorchSharp native dlls. See <see cref="T:Microsoft.ML.Vision.ImageClassificationTrainer"/> for more information.
    /// </remarks>
    public static class TorchSharpCatalog
    {
        [System.Diagnostics.CodeAnalysis.SuppressMessage("Naming", "MSML_GeneralName:This name should be PascalCased", Justification = "<Pending>")]
        public static MNISTTrainer MNIST(
    this MulticlassClassificationCatalog.MulticlassClassificationTrainers catalog,
    string labelColumn = DefaultColumnNames.Label,
            string featureColumn = DefaultColumnNames.Features,
            string scoreColumn = DefaultColumnNames.Score,
            string predictedLabelColumn = DefaultColumnNames.PredictedLabel,
            IDataView validationSet = null) =>
        new MNISTTrainer(CatalogUtils.GetEnvironment(catalog), labelColumn, featureColumn, scoreColumn, predictedLabelColumn, validationSet);
    }
}

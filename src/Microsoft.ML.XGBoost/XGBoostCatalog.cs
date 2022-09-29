// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.



using System;
using System.Collections.Generic;
using System.Text;

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.XGBoost;

namespace Microsoft.ML
{
    /// <summary>
    /// Collection of extension methods for the <see cref="RegressionCatalog.RegressionTrainers"/>,
    ///  <see cref="BinaryClassificationCatalog.BinaryClassificationTrainers"/>, 
    ///  and <see cref="MulticlassClassificationCatalog.MulticlassClassificationTrainers"/> catalogs.
    /// </summary>
    public static class XGBoostExtensions
    {
        public static XGBoostBinaryClassificationEstimator XGBoost(this BinaryClassificationCatalog.BinaryClassificationTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Defaults.NumberOfIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new XGBoostBinaryClassificationEstimator(env, labelColumnName, featureColumnName, numberOfLeaves, minimumExampleCountPerLeaf, learningRate, numberOfIterations);
        }
    }
}

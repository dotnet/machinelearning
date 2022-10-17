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
    /// Collection of extension methods for the <see cref="RegressionCatalog.RegressionTrainers"/> and
    ///  <see cref="BinaryClassificationCatalog.BinaryClassificationTrainers"/> catalogs.
    /// </summary>
    public static class XGBoostExtensions
    {
        /// <summary>
        /// Create <see cref="XGBoostRegressionTrainer"/>, which predicts a target using a gradient boosting decision tree regression model.
        /// </summary>
        /// <param name="catalog">The <see cref="RegressionCatalog"/>.</param>
        /// <param name="labelColumnName">The name of the label column. The column data must be <see cref="System.Single"/>.</param>
        /// <param name="featureColumnName">The name of the feature column. The column data must be a known-sized vector of <see cref="System.Single"/>.</param>
        /// <param name="exampleWeightColumnName">The name of the example weight column (optional).</param>
        /// <param name="numberOfLeaves">The maximum number of leaves in one tree.</param>
        /// <param name="minimumExampleCountPerLeaf">The minimal number of data points required to form a new tree leaf.</param>
        /// <param name="learningRate">The learning rate.</param>
        /// <param name="numberOfIterations">The number of boosting iterations. A new tree is created in each iteration, so this is equivalent to the number of trees.</param>
        public static XGBoostRegressionTrainer LightGbm(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null,
            int? numberOfLeaves = null,
            int? minimumExampleCountPerLeaf = null,
            double? learningRate = null,
            int numberOfIterations = Defaults.NumberOfIterations)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new XGBoostRegressionTrainer(env, labelColumnName, featureColumnName, exampleWeightColumnName, numberOfLeaves, minimumExampleCountPerLeaf, learningRate, numberOfIterations);
        }

        public static XGBoostRegressionTrainer XGBoost(this RegressionCatalog.RegressionTrainers catalog,
            XGBoostRegressionTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            return new XGBoostRegressionTrainer(env, options);
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.AutoML;
using Microsoft.ML.Fairlearn.reductions;

namespace Microsoft.ML.Fairlearn.AutoML
{
    /// <summary>
    /// An internal class that holds the gridLimit value to conduct gridsearch.
    /// Needed to pass the value into the AutoMLExperiment as a singleton
    /// </summary>
    internal class GridLimit
    {
        public float Value { get; set; }
    }
    /// <summary>
    /// An extension class used to add more options to the Fairlearn girdsearch experiment
    /// </summary>
    public static class AutoMLExperimentExtension
    {
        public static AutoMLExperiment SetBinaryClassificationMoment(this AutoMLExperiment experiment, ClassificationMoment moment)
        {
            experiment.ServiceCollection.AddSingleton(moment);

            return experiment;
        }

        public static AutoMLExperiment SetGridLimit(this AutoMLExperiment experiment, float gridLimit)
        {
            var gridLimitObject = new GridLimit();
            gridLimitObject.Value = gridLimit;
            experiment.ServiceCollection.AddSingleton(gridLimitObject);
            experiment.SetTuner<CostFrugalWithLambdaTunerFactory>();

            return experiment;
        }
    }
}

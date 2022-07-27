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
    public static class AutoMLExperimentExtension
    {
        public static AutoMLExperiment SetBinaryClassificationMoment(this AutoMLExperiment experiment, ClassificationMoment moment)
        {
            experiment.ServiceCollection.AddSingleton(moment);
            experiment.SetTunerFactory<CostFrugalWithLambdaTunerFactory>();

            return experiment;
        }
    }
}

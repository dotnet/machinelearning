// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.SearchSpace;

namespace Microsoft.ML.AutoML
{
    internal class TrialSettings
    {
        public int TrialId { get; set; }

        public SweepableEstimatorPipeline Pipeline { get; set; }

        public string Schema { get; set; }

        public Parameter Parameter { get; set; }

        public AutoMLExperiment.AutoMLExperimentSettings ExperimentSettings { get; set; }
    }
}

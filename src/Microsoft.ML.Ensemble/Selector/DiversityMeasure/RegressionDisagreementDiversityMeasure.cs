﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.Ensemble.Selector;
using Microsoft.ML.Ensemble.Selector.DiversityMeasure;

[assembly: LoadableClass(typeof(RegressionDisagreementDiversityMeasure), null, typeof(SignatureEnsembleDiversityMeasure),
    DisagreementDiversityMeasure.UserName, RegressionDisagreementDiversityMeasure.LoadName)]

namespace Microsoft.ML.Ensemble.Selector.DiversityMeasure
{
    public class RegressionDisagreementDiversityMeasure : BaseDisagreementDiversityMeasure<Single>, IRegressionDiversityMeasure
    {
        public const string LoadName = "RegressionDisagreementDiversityMeasure";

        protected override Single GetDifference(in Single valueX, in Single valueY)
        {
            return Math.Abs(valueX - valueY);
        }
    }
}

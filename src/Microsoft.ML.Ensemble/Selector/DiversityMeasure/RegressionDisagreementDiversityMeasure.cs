// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Ensemble.Selector.DiversityMeasure;

[assembly: LoadableClass(typeof(RegressionDisagreementDiversityMeasure), null, typeof(SignatureEnsembleDiversityMeasure),
    DisagreementDiversityMeasure.UserName, RegressionDisagreementDiversityMeasure.LoadName)]

namespace Microsoft.ML.Runtime.Ensemble.Selector.DiversityMeasure
{
    public class RegressionDisagreementDiversityMeasure : BaseDisagreementDiversityMeasure<Single>
    {
        public const string LoadName = "RegressionDisagreementDiversityMeasure";

        protected override Single GetDifference(ref Single valueX, ref Single valueY)
        {
            return Math.Abs(valueX - valueY);
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Ensemble.Selector.DiversityMeasure;

[assembly: LoadableClass(typeof(DisagreementDiversityMeasure), null, typeof(SignatureEnsembleDiversityMeasure),
    DisagreementDiversityMeasure.UserName, DisagreementDiversityMeasure.LoadName)]

namespace Microsoft.ML.Runtime.Ensemble.Selector.DiversityMeasure
{
    public class DisagreementDiversityMeasure : BaseDisagreementDiversityMeasure<Single>, IBinaryDiversityMeasure
    {
        public const string UserName = "Disagreement Diversity Measure";
        public const string LoadName = "DisagreementDiversityMeasure";

        protected override Single GetDifference(ref Single valueX, ref Single valueY)
        {
            return (valueX > 0 && valueY < 0 || valueX < 0 && valueY > 0) ? 1 : 0;
        }
    }
}

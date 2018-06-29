// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Ensemble.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Ensemble.Selector.DiversityMeasure;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: EntryPointModule(typeof(DisagreementDiversityFactory))]
[assembly: EntryPointModule(typeof(RegressionDisagreementDiversityFactory))]
[assembly: EntryPointModule(typeof(MultiDisagreementDiversityFactory))]

namespace Microsoft.ML.Ensemble.EntryPoints
{
    [TlcModule.Component(Name = DisagreementDiversityMeasure.LoadName, FriendlyName = DisagreementDiversityMeasure.UserName)]
    public sealed class DisagreementDiversityFactory : ISupportDiversityMeasureFactory<Single>
    {
        public IDiversityMeasure<float> CreateComponent(IHostEnvironment env) => new DisagreementDiversityMeasure();
    }

    [TlcModule.Component(Name = RegressionDisagreementDiversityMeasure.LoadName, FriendlyName = DisagreementDiversityMeasure.UserName)]
    public sealed class RegressionDisagreementDiversityFactory : ISupportDiversityMeasureFactory<Single>
    {
        public IDiversityMeasure<float> CreateComponent(IHostEnvironment env) => new RegressionDisagreementDiversityMeasure();
    }

    [TlcModule.Component(Name = MultiDisagreementDiversityMeasure.LoadName, FriendlyName = DisagreementDiversityMeasure.UserName)]
    public sealed class MultiDisagreementDiversityFactory : ISupportDiversityMeasureFactory<VBuffer<Single>>
    {
        public IDiversityMeasure<VBuffer<Single>> CreateComponent(IHostEnvironment env) => new MultiDisagreementDiversityMeasure();
    }

}

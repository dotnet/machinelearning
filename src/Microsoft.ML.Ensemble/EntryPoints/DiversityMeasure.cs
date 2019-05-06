// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Ensemble;

[assembly: EntryPointModule(typeof(DisagreementDiversityFactory))]
[assembly: EntryPointModule(typeof(RegressionDisagreementDiversityFactory))]
[assembly: EntryPointModule(typeof(MultiDisagreementDiversityFactory))]

namespace Microsoft.ML.Trainers.Ensemble
{
    [TlcModule.Component(Name = DisagreementDiversityMeasure.LoadName, FriendlyName = DisagreementDiversityMeasure.UserName)]
    internal sealed class DisagreementDiversityFactory : ISupportBinaryDiversityMeasureFactory
    {
        public IBinaryDiversityMeasure CreateComponent(IHostEnvironment env) => new DisagreementDiversityMeasure();
    }

    [TlcModule.Component(Name = RegressionDisagreementDiversityMeasure.LoadName, FriendlyName = DisagreementDiversityMeasure.UserName)]
    internal sealed class RegressionDisagreementDiversityFactory : ISupportRegressionDiversityMeasureFactory
    {
        public IRegressionDiversityMeasure CreateComponent(IHostEnvironment env) => new RegressionDisagreementDiversityMeasure();
    }

    [TlcModule.Component(Name = MultiDisagreementDiversityMeasure.LoadName, FriendlyName = DisagreementDiversityMeasure.UserName)]
    internal sealed class MultiDisagreementDiversityFactory : ISupportMulticlassDiversityMeasureFactory
    {
        public IMulticlassDiversityMeasure CreateComponent(IHostEnvironment env) => new MultiDisagreementDiversityMeasure();
    }

}

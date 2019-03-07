// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Ensemble;
using Microsoft.ML.Trainers.Ensemble.SubsetSelector;

[assembly: LoadableClass(typeof(AllInstanceSelector), typeof(AllInstanceSelector.Arguments),
    typeof(SignatureEnsembleDataSelector), AllInstanceSelector.UserName, AllInstanceSelector.LoadName)]

[assembly: EntryPointModule(typeof(AllInstanceSelector))]

namespace Microsoft.ML.Trainers.Ensemble.SubsetSelector
{
    internal sealed class AllInstanceSelector : BaseSubsetSelector<AllInstanceSelector.Arguments>
    {
        public const string UserName = "All Instance Selector";
        public const string LoadName = "AllInstanceSelector";

        [TlcModule.Component(Name = LoadName, FriendlyName = UserName)]
        public sealed class Arguments : ArgumentsBase, ISupportSubsetSelectorFactory
        {
            public ISubsetSelector CreateComponent(IHostEnvironment env) => new AllInstanceSelector(env, this);
        }

        public AllInstanceSelector(IHostEnvironment env, Arguments args)
            : base(args, env, LoadName)
        {
        }

        public override IEnumerable<Subset> GetSubsets(Batch batch, Random rand)
        {
            for (int i = 0; i < Size; i++)
                yield return FeatureSelector.SelectFeatures(batch.TrainInstances, rand);
        }
    }
}

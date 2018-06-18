// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Ensemble.Selector.SubsetSelector;

[assembly: LoadableClass(typeof(AllInstanceSelector), typeof(AllInstanceSelector.Arguments),
    typeof(SignatureEnsembleDataSelector), AllInstanceSelector.UserName, AllInstanceSelector.LoadName)]

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubsetSelector
{
    public sealed class AllInstanceSelector : BaseSubsetSelector<AllInstanceSelector.Arguments>
    {
        public const string UserName = "All Instance Selector";
        public const string LoadName = "AllInstanceSelector";

        public sealed class Arguments : ArgumentsBase
        {
        }

        public AllInstanceSelector(IHostEnvironment env, Arguments args)
            : base(args, env, LoadName)
        {
        }

        public override IEnumerable<Subset> GetSubsets(Batch batch, IRandom rand)
        {
            for (int i = 0; i < Size; i++)
                yield return FeatureSelector.SelectFeatures(batch.TrainInstances, rand);
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Ensemble.Selector.SubsetSelector;
using Microsoft.ML.Runtime.EntryPoints;

[assembly: LoadableClass(typeof(BootstrapSelector), typeof(BootstrapSelector.Arguments),
    typeof(SignatureEnsembleDataSelector), BootstrapSelector.UserName, BootstrapSelector.LoadName)]

[assembly: EntryPointModule(typeof(BootstrapSelector))]

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubsetSelector
{
    public sealed class BootstrapSelector : BaseSubsetSelector<BootstrapSelector.Arguments>
    {
        public const string UserName = "Bootstrap Selector";
        public const string LoadName = "BootstrapSelector";

        [TlcModule.Component(Name = LoadName, FriendlyName = UserName)]
        public sealed class Arguments : ArgumentsBase, ISupportSubsetSelectorFactory
        {
            // REVIEW: This could be reintroduced by having the transform counting the
            // proportions of each label, then adjusting the lambdas accordingly. However, at
            // the current point in time supporting this non-default action is not considered
            // a priority.
#if OLD_ENSEMBLE
            [Argument(ArgumentType.AtMostOnce, HelpText = "If checked, the classes will be balanced by over sampling of minority classes", ShortName = "cb", SortOrder = 50)]
            public bool balanced = false;
#endif
            public ISubsetSelector CreateComponent(IHostEnvironment env) => new BootstrapSelector(env, this);
        }

        public BootstrapSelector(IHostEnvironment env, Arguments args)
            : base(args, env, LoadName)
        {
        }

        public override IEnumerable<Subset> GetSubsets(Batch batch, IRandom rand)
        {
            for (int i = 0; i < Size; i++)
            {
                // REVIEW: Consider ways to reintroduce "balanced" samples.
                var viewTrain = new BootstrapSampleTransform(Host, new BootstrapSampleTransform.Arguments(), Data.Data);
                var dataTrain = RoleMappedData.Create(viewTrain, Data.Schema.GetColumnRoleNames());
                yield return FeatureSelector.SelectFeatures(dataTrain, rand);
            }
        }
    }
}

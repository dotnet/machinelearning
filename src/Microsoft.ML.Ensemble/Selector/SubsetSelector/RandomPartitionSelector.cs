// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble.Selector;
using Microsoft.ML.Runtime.Ensemble.Selector.SubsetSelector;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;

[assembly: LoadableClass(typeof(RandomPartitionSelector), typeof(RandomPartitionSelector.Arguments),
    typeof(SignatureEnsembleDataSelector), RandomPartitionSelector.UserName, RandomPartitionSelector.LoadName)]

[assembly: EntryPointModule(typeof(RandomPartitionSelector))]

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubsetSelector
{
    public sealed class RandomPartitionSelector : BaseSubsetSelector<RandomPartitionSelector.Arguments>
    {
        public const string UserName = "Random Partition Selector";
        public const string LoadName = "RandomPartitionSelector";

        [TlcModule.Component(Name = LoadName, FriendlyName = UserName)]
        public sealed class Arguments : ArgumentsBase, ISupportSubsetSelectorFactory
        {
            public ISubsetSelector CreateComponent(IHostEnvironment env) => new RandomPartitionSelector(env, this);
        }

        public RandomPartitionSelector(IHostEnvironment env, Arguments args)
            : base(args, env, LoadName)
        {
        }

        public override IEnumerable<Subset> GetSubsets(Batch batch, IRandom rand)
        {
            string name = Data.Data.Schema.GetTempColumnName();
            var args = new NumberGeneratingTransformer.Arguments();
            args.Column = new[] { new NumberGeneratingTransformer.Column() { Name = name } };
            args.Seed = (uint)rand.Next();
            IDataTransform view = new NumberGeneratingTransformer(Host, args, Data.Data);

            // REVIEW: This won't be very efficient when Size is large.
            for (int i = 0; i < Size; i++)
            {
                var viewTrain = new RangeFilter(Host, new RangeFilter.Arguments() { Column = name, Min = (Double)i / Size, Max = (Double)(i + 1) / Size }, view);
                var dataTrain = new RoleMappedData(viewTrain, Data.Schema.GetColumnRoleNames());
                yield return FeatureSelector.SelectFeatures(dataTrain, rand);
            }
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers.Ensemble;

[assembly: LoadableClass(typeof(RandomFeatureSelector), typeof(RandomFeatureSelector.Arguments),
    typeof(SignatureEnsembleFeatureSelector), RandomFeatureSelector.UserName, RandomFeatureSelector.LoadName)]

namespace Microsoft.ML.Trainers.Ensemble
{
    internal class RandomFeatureSelector : IFeatureSelector
    {
        public const string UserName = "Random Feature Selector";
        public const string LoadName = "RandomFeatureSelector";

        [TlcModule.Component(Name = LoadName, FriendlyName = UserName)]
        public sealed class Arguments : ISupportFeatureSelectorFactory
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The proportion of features to be selected. The range is 0.0-1.0", ShortName = "fp", SortOrder = 50)]
            public Single FeaturesSelectionProportion = 0.8f;

            public IFeatureSelector CreateComponent(IHostEnvironment env) => new RandomFeatureSelector(env, this);
        }

        private readonly Arguments _args;
        private readonly IHost _host;

        public RandomFeatureSelector(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));

            _host = env.Register(LoadName);
            _args = args;
            _host.Check(0 < _args.FeaturesSelectionProportion && _args.FeaturesSelectionProportion < 1,
                "The feature proportion for RandomFeatureSelector should be greater than 0 and lesser than 1");
        }

        public Subset SelectFeatures(RoleMappedData data, Random rand)
        {
            _host.CheckValue(data, nameof(data));
            data.CheckFeatureFloatVector();

            var type = data.Schema.Feature.Value.Type;
            int len = type.GetVectorSize();
            var features = new BitArray(len);
            for (int j = 0; j < len; j++)
                features[j] = rand.NextDouble() < _args.FeaturesSelectionProportion;
            var dataNew = EnsembleUtils.SelectFeatures(_host, data, features);
            return new Subset(dataNew, features);
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Ensemble.EntryPoints;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Ensemble.Selector.SubsetSelector
{
    public abstract class BaseSubsetSelector<TArgs> : ISubsetSelector
        where TArgs : BaseSubsetSelector<TArgs>.ArgumentsBase
    {
        public abstract class ArgumentsBase
        {
            [Argument(ArgumentType.Multiple, HelpText = "The Feature selector", ShortName = "fs", SortOrder = 1)]
            public ISupportFeatureSelectorFactory FeatureSelector = new AllFeatureSelectorFactory();
        }

        protected readonly IHost Host;
        protected readonly TArgs Args;
        protected readonly IFeatureSelector FeatureSelector;

        protected int Size;
        protected RoleMappedData Data;
        protected int BatchSize;
        protected Single ValidationDatasetProportion;

        protected BaseSubsetSelector(TArgs args, IHostEnvironment env, string name)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(args, nameof(args));
            env.CheckNonWhiteSpace(name, nameof(name));

            Host = env.Register(name);
            Args = args;
            FeatureSelector = Args.FeatureSelector.CreateComponent(Host);
        }

        public void Initialize(RoleMappedData data, int size, int batchSize, Single validationDatasetProportion)
        {
            Host.CheckValue(data, nameof(data));
            Host.CheckParam(size > 0, nameof(size));
            Host.CheckParam(0 <= validationDatasetProportion && validationDatasetProportion < 1,
                nameof(validationDatasetProportion), "Should be greater than or equal to 0 and less than 1");
            Data = data;
            Size = size;
            BatchSize = batchSize;
            ValidationDatasetProportion = validationDatasetProportion;
        }

        public abstract IEnumerable<Subset> GetSubsets(Batch batch, IRandom rand);

        public IEnumerable<Batch> GetBatches(IRandom rand)
        {
            Host.Assert(Data != null, "Must call Initialize first!");
            Host.AssertValue(rand);

            using (var ch = Host.Start("Getting batches"))
            {
                RoleMappedData dataTest;
                RoleMappedData dataTrain;

                // Split the data, if needed.
                if (!(ValidationDatasetProportion > 0))
                    dataTest = dataTrain = Data;
                else
                {
                    // Split the data into train and test sets.
                    string name = Data.Data.Schema.GetTempColumnName();
                    var args = new GenerateNumberTransform.Arguments();
                    args.Column = new[] { new GenerateNumberTransform.Column() { Name = name } };
                    args.Seed = (uint)rand.Next();
                    var view = new GenerateNumberTransform(Host, args, Data.Data);
                    var viewTest = new RangeFilter(Host, new RangeFilter.Arguments() { Column = name, Max = ValidationDatasetProportion }, view);
                    var viewTrain = new RangeFilter(Host, new RangeFilter.Arguments() { Column = name, Max = ValidationDatasetProportion, Complement = true }, view);
                    dataTest = RoleMappedData.Create(viewTest, Data.Schema.GetColumnRoleNames());
                    dataTrain = RoleMappedData.Create(viewTrain, Data.Schema.GetColumnRoleNames());
                }

                if (BatchSize > 0)
                {
                    // REVIEW: How should we carve the data into batches?
                    ch.Warning("Batch support is temporarily disabled");
                }

                yield return new Batch(dataTrain, dataTest);
                ch.Done();
            }
        }

        public virtual RoleMappedData GetTestData(Subset subset, Batch batch)
        {
            Host.CheckValueOrNull(subset);
            Host.CheckValue(batch.TestInstances, nameof(batch), "Batch does not have test data");

            if (subset == null || subset.SelectedFeatures == null)
                return batch.TestInstances;
            return EnsembleUtils.SelectFeatures(Host, batch.TestInstances, subset.SelectedFeatures);
        }
    }
}

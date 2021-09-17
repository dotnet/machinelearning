// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(void), typeof(TrainTestSplit), null, typeof(SignatureEntryPointModule), "TrainTestSplit")]

namespace Microsoft.ML.EntryPoints
{
    internal static class TrainTestSplit
    {
        public sealed class Input
        {
            [Argument(ArgumentType.Required, HelpText = "Input dataset", SortOrder = 1)]
            public IDataView Data;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Fraction of training data", SortOrder = 2)]
            public float Fraction = 0.8f;

            [Argument(ArgumentType.AtMostOnce, ShortName = "strat", HelpText = "Stratification column", SortOrder = 3)]
            public string StratificationColumn = null;
        }

        public sealed class Output
        {
            [TlcModule.Output(Desc = "Training data", SortOrder = 1)]
            public IDataView TrainData;

            [TlcModule.Output(Desc = "Testing data", SortOrder = 2)]
            public IDataView TestData;
        }

        public const string ModuleName = "TrainTestSplit";
        public const string UserName = "Dataset Train-Test Split";

        [TlcModule.EntryPoint(Name = "Transforms.TrainTestDatasetSplitter", Desc = "Split the dataset into train and test sets", UserName = UserName)]
        public static Output Split(IHostEnvironment env, Input input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(ModuleName);
            host.CheckValue(input, nameof(input));
            host.Check(0 < input.Fraction && input.Fraction < 1, "The fraction must be in the interval (0,1).");

            EntryPointUtils.CheckInputArgs(host, input);

            var data = input.Data;
            var splitCol = DataOperationsCatalog.CreateSplitColumn(env, ref data, input.StratificationColumn);

            IDataView trainData = new RangeFilter(host,
                new RangeFilter.Options { Column = splitCol, Min = 0, Max = input.Fraction, Complement = false }, data);
            trainData = ColumnSelectingTransformer.CreateDrop(host, trainData, splitCol);

            IDataView testData = new RangeFilter(host,
                new RangeFilter.Options { Column = splitCol, Min = 0, Max = input.Fraction, Complement = true }, data);
            testData = ColumnSelectingTransformer.CreateDrop(host, testData, splitCol);

            return new Output() { TrainData = trainData, TestData = testData };
        }

    }
}

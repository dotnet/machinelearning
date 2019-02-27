// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.EntryPoints;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(void), typeof(CVSplit), null, typeof(SignatureEntryPointModule), "CVSplit")]

namespace Microsoft.ML.EntryPoints
{
    /// <summary>
    /// The module that splits the input dataset into the specified number of cross-validation folds, and outputs the 'training'
    /// and 'testing' portion of the input for each fold.
    /// </summary>
    internal static class CVSplit
    {
        public sealed class Input
        {
            [Argument(ArgumentType.Required, HelpText = "Input dataset", SortOrder = 1)]
            public IDataView Data;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of folds to split into", SortOrder = 2)]
            public int NumFolds = 2;

            [Argument(ArgumentType.AtMostOnce, ShortName = "strat", HelpText = "Stratification column", SortOrder = 3)]
            public string StratificationColumn = null;
        }

        public sealed class Output
        {
            [TlcModule.Output(Desc = "Training data (one dataset per fold)", SortOrder = 1)]
            public IDataView[] TrainData;

            [TlcModule.Output(Desc = "Testing data (one dataset per fold)", SortOrder = 2)]
            public IDataView[] TestData;
        }

        public const string ModuleName = "CVSplit";
        public const string UserName = "Dataset CV Split";

        [TlcModule.EntryPoint(Name = "Models.CrossValidatorDatasetSplitter", Desc = "Split the dataset into the specified number of cross-validation folds (train and test sets)", UserName = UserName)]
        public static Output Split(IHostEnvironment env, Input input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(ModuleName);
            host.CheckValue(input, nameof(input));

            EntryPointUtils.CheckInputArgs(host, input);

            var data = input.Data;

            var stratCol = SplitUtils.CreateStratificationColumn(host, ref data, input.StratificationColumn);

            int n = input.NumFolds;
            var output = new Output
            {
                TrainData = new IDataView[n],
                TestData = new IDataView[n]
            };

            // Construct per-fold datasets.
            double fraction = 1.0 / n;
            for (int i = 0; i < n; i++)
            {
                var trainData = new RangeFilter(host,
                    new RangeFilter.Options { Column = stratCol, Min = i * fraction, Max = (i + 1) * fraction, Complement = true }, data);
                output.TrainData[i] = ColumnSelectingTransformer.CreateDrop(host, trainData, stratCol);

                var testData = new RangeFilter(host,
                    new RangeFilter.Options { Column = stratCol, Min = i * fraction, Max = (i + 1) * fraction, Complement = false }, data);
                output.TestData[i] = ColumnSelectingTransformer.CreateDrop(host, testData, stratCol);
            }

            return output;
        }
    }
}

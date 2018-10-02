// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using Microsoft.ML.Ensemble.EntryPoints;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Ensemble;
using Microsoft.ML.Runtime.Ensemble.OutputCombiners;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;

[assembly: LoadableClass(typeof(void), typeof(EnsembleCreator), null, typeof(SignatureEntryPointModule), "CreateEnsemble")]

namespace Microsoft.ML.Runtime.EntryPoints
{
    /// <summary>
    /// A component to combine given models into an ensemble model.
    /// </summary>
    public static class EnsembleCreator
    {
        /// <summary>
        /// These are the combiner options for binary and multi class classifiers.
        /// </summary>
        public enum ClassifierCombiner
        {
            Median,
            Average,
            Vote,
        }

        /// <summary>
        /// These are the combiner options for regression and anomaly detection.
        /// </summary>
        public enum ScoreCombiner
        {
            Median,
            Average,
        }

        public abstract class PipelineInputBase
        {
            [Argument(ArgumentType.Required, ShortName = "models", HelpText = "The models to combine into an ensemble", SortOrder = 1)]
            public IPredictorModel[] Models;
        }

        public abstract class InputBase
        {
            [Argument(ArgumentType.Required, ShortName = "models", HelpText = "The models to combine into an ensemble", SortOrder = 1)]
            public IPredictorModel[] Models;

            [Argument(ArgumentType.AtMostOnce, ShortName = "validate", HelpText = "Whether to validate that all the pipelines are identical", SortOrder = 5)]
            public bool ValidatePipelines = true;
        }

        public sealed class ClassifierInput : InputBase
        {
            [Argument(ArgumentType.AtMostOnce, ShortName = "combiner", HelpText = "The combiner used to combine the scores", SortOrder = 2)]
            public ClassifierCombiner ModelCombiner = ClassifierCombiner.Median;
        }

        public sealed class PipelineClassifierInput : PipelineInputBase
        {
            [Argument(ArgumentType.AtMostOnce, ShortName = "combiner", HelpText = "The combiner used to combine the scores", SortOrder = 2)]
            public ClassifierCombiner ModelCombiner = ClassifierCombiner.Median;
        }

        public sealed class RegressionInput : InputBase
        {
            [Argument(ArgumentType.AtMostOnce, ShortName = "combiner", HelpText = "The combiner used to combine the scores", SortOrder = 2)]
            public ScoreCombiner ModelCombiner = ScoreCombiner.Median;
        }

        public sealed class PipelineRegressionInput : PipelineInputBase
        {
            [Argument(ArgumentType.AtMostOnce, ShortName = "combiner", HelpText = "The combiner used to combine the scores", SortOrder = 2)]
            public ScoreCombiner ModelCombiner = ScoreCombiner.Median;
        }

        public sealed class PipelineAnomalyInput : PipelineInputBase
        {
            [Argument(ArgumentType.AtMostOnce, ShortName = "combiner", HelpText = "The combiner used to combine the scores", SortOrder = 2)]
            public ScoreCombiner ModelCombiner = ScoreCombiner.Average;
        }

        private static void GetPipeline(IHostEnvironment env, InputBase input, out IDataView startingData, out RoleMappedData transformedData)
        {
            Contracts.AssertValue(env);
            env.AssertValue(input);
            env.AssertNonEmpty(input.Models);

            ISchema inputSchema = null;
            startingData = null;
            transformedData = null;
            byte[][] transformedDataSerialized = null;
            string[] transformedDataZipEntryNames = null;
            for (int i = 0; i < input.Models.Length; i++)
            {
                var model = input.Models[i];

                var inputData = new EmptyDataView(env, model.TransformModel.InputSchema);
                model.PrepareData(env, inputData, out RoleMappedData transformedDataCur, out IPredictor pred);

                if (inputSchema == null)
                {
                    env.Assert(i == 0);
                    inputSchema = model.TransformModel.InputSchema;
                    startingData = inputData;
                    transformedData = transformedDataCur;
                }
                else if (input.ValidatePipelines)
                {
                    using (var ch = env.Start("Validating pipeline"))
                    {
                        if (transformedDataSerialized == null)
                        {
                            ch.Assert(transformedDataZipEntryNames == null);
                            SerializeRoleMappedData(env, ch, transformedData, out transformedDataSerialized,
                                out transformedDataZipEntryNames);
                        }
                        CheckSamePipeline(env, ch, transformedDataCur, transformedDataSerialized, transformedDataZipEntryNames);
                        ch.Done();
                    }
                }
            }
        }

        [TlcModule.EntryPoint(Name = "Models.BinaryEnsemble", Desc = "Combine binary classifiers into an ensemble", UserName = EnsembleTrainer.UserNameValue)]
        public static CommonOutputs.BinaryClassificationOutput CreateBinaryEnsemble(IHostEnvironment env, ClassifierInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CombineModels");
            host.CheckValue(input, nameof(input));
            host.CheckNonEmpty(input.Models, nameof(input.Models));

            GetPipeline(host, input, out IDataView startingData, out RoleMappedData transformedData);

            var args = new EnsembleTrainer.Arguments();
            switch (input.ModelCombiner)
            {
                case ClassifierCombiner.Median:
                    args.OutputCombiner = new MedianFactory();
                    break;
                case ClassifierCombiner.Average:
                    args.OutputCombiner = new AverageFactory();
                    break;
                case ClassifierCombiner.Vote:
                    args.OutputCombiner = new VotingFactory();
                    break;
                default:
                    throw host.Except("Unknown combiner kind");
            }

            var trainer = new EnsembleTrainer(host, args);
            var ensemble = trainer.CombineModels(input.Models.Select(pm => pm.Predictor as IPredictorProducing<float>));

            var predictorModel = new PredictorModel(host, transformedData, startingData, ensemble);

            var output = new CommonOutputs.BinaryClassificationOutput { PredictorModel = predictorModel };
            return output;
        }

        [TlcModule.EntryPoint(Name = "Models.RegressionEnsemble", Desc = "Combine regression models into an ensemble", UserName = RegressionEnsembleTrainer.UserNameValue)]
        public static CommonOutputs.RegressionOutput CreateRegressionEnsemble(IHostEnvironment env, RegressionInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CombineModels");
            host.CheckValue(input, nameof(input));
            host.CheckNonEmpty(input.Models, nameof(input.Models));

            GetPipeline(host, input, out IDataView startingData, out RoleMappedData transformedData);

            var args = new RegressionEnsembleTrainer.Arguments();
            switch (input.ModelCombiner)
            {
                case ScoreCombiner.Median:
                    args.OutputCombiner = new MedianFactory();
                    break;
                case ScoreCombiner.Average:
                    args.OutputCombiner = new AverageFactory();
                    break;
                default:
                    throw host.Except("Unknown combiner kind");
            }

            var trainer = new RegressionEnsembleTrainer(host, args);
            var ensemble = trainer.CombineModels(input.Models.Select(pm => pm.Predictor as IPredictorProducing<float>));

            var predictorModel = new PredictorModel(host, transformedData, startingData, ensemble);

            var output = new CommonOutputs.RegressionOutput { PredictorModel = predictorModel };
            return output;
        }

        [TlcModule.EntryPoint(Name = "Models.BinaryPipelineEnsemble", Desc = "Combine binary classification models into an ensemble")]
        public static CommonOutputs.BinaryClassificationOutput CreateBinaryPipelineEnsemble(IHostEnvironment env, PipelineClassifierInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CombineModels");
            host.CheckValue(input, nameof(input));
            host.CheckNonEmpty(input.Models, nameof(input.Models));

            IBinaryOutputCombiner combiner;
            switch (input.ModelCombiner)
            {
                case ClassifierCombiner.Median:
                    combiner = new Median(host);
                    break;
                case ClassifierCombiner.Average:
                    combiner = new Average(host);
                    break;
                case ClassifierCombiner.Vote:
                    combiner = new Voting(host);
                    break;
                default:
                    throw host.Except("Unknown combiner kind");
            }
            var ensemble = SchemaBindablePipelineEnsembleBase.Create(host, input.Models, combiner, MetadataUtils.Const.ScoreColumnKind.BinaryClassification);
            return CreatePipelineEnsemble<CommonOutputs.BinaryClassificationOutput>(host, input.Models, ensemble);
        }

        [TlcModule.EntryPoint(Name = "Models.RegressionPipelineEnsemble", Desc = "Combine regression models into an ensemble")]
        public static CommonOutputs.RegressionOutput CreateRegressionPipelineEnsemble(IHostEnvironment env, PipelineRegressionInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CombineModels");
            host.CheckValue(input, nameof(input));
            host.CheckNonEmpty(input.Models, nameof(input.Models));

            IRegressionOutputCombiner combiner;
            switch (input.ModelCombiner)
            {
                case ScoreCombiner.Median:
                    combiner = new Median(host);
                    break;
                case ScoreCombiner.Average:
                    combiner = new Average(host);
                    break;
                default:
                    throw host.Except("Unknown combiner kind");
            }
            var ensemble = SchemaBindablePipelineEnsembleBase.Create(host, input.Models, combiner, MetadataUtils.Const.ScoreColumnKind.Regression);
            return CreatePipelineEnsemble<CommonOutputs.RegressionOutput>(host, input.Models, ensemble);
        }

        [TlcModule.EntryPoint(Name = "Models.MultiClassPipelineEnsemble", Desc = "Combine multiclass classifiers into an ensemble")]
        public static CommonOutputs.MulticlassClassificationOutput CreateMultiClassPipelineEnsemble(IHostEnvironment env, PipelineClassifierInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CombineModels");
            host.CheckValue(input, nameof(input));
            host.CheckNonEmpty(input.Models, nameof(input.Models));

            IOutputCombiner<VBuffer<Single>> combiner;
            switch (input.ModelCombiner)
            {
                case ClassifierCombiner.Median:
                    combiner = new MultiMedian(host, new MultiMedian.Arguments() { Normalize = true });
                    break;
                case ClassifierCombiner.Average:
                    combiner = new MultiAverage(host, new MultiAverage.Arguments() { Normalize = true });
                    break;
                case ClassifierCombiner.Vote:
                    combiner = new MultiVoting(host);
                    break;
                default:
                    throw host.Except("Unknown combiner kind");
            }
            var ensemble = SchemaBindablePipelineEnsembleBase.Create(host, input.Models, combiner, MetadataUtils.Const.ScoreColumnKind.MultiClassClassification);
            return CreatePipelineEnsemble<CommonOutputs.MulticlassClassificationOutput>(host, input.Models, ensemble);
        }

        [TlcModule.EntryPoint(Name = "Models.AnomalyPipelineEnsemble", Desc = "Combine anomaly detection models into an ensemble")]
        public static CommonOutputs.AnomalyDetectionOutput CreateAnomalyPipelineEnsemble(IHostEnvironment env, PipelineAnomalyInput input)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register("CombineModels");
            host.CheckValue(input, nameof(input));
            host.CheckNonEmpty(input.Models, nameof(input.Models));

            IRegressionOutputCombiner combiner;
            switch (input.ModelCombiner)
            {
                case ScoreCombiner.Median:
                    combiner = new Median(host);
                    break;
                case ScoreCombiner.Average:
                    combiner = new Average(host);
                    break;
                default:
                    throw host.Except("Unknown combiner kind");
            }
            var ensemble = SchemaBindablePipelineEnsembleBase.Create(host, input.Models, combiner, MetadataUtils.Const.ScoreColumnKind.AnomalyDetection);
            return CreatePipelineEnsemble<CommonOutputs.AnomalyDetectionOutput>(host, input.Models, ensemble);
        }

        private static TOut CreatePipelineEnsemble<TOut>(IHostEnvironment env, IPredictorModel[] predictors, SchemaBindablePipelineEnsembleBase ensemble)
            where TOut : CommonOutputs.TrainerOutput, new()
        {
            var inputSchema = predictors[0].TransformModel.InputSchema;
            var dv = new EmptyDataView(env, inputSchema);

            // The role mappings are specific to the individual predictors.
            var rmd = new RoleMappedData(dv);
            var predictorModel = new PredictorModel(env, rmd, dv, ensemble);

            var output = new TOut { PredictorModel = predictorModel };
            return output;
        }

        /// <summary>
        /// This method takes a <see cref="RoleMappedData"/> as input, saves it as an in-memory <see cref="ZipArchive"/>
        /// and returns two arrays indexed by the entries in the zip:
        /// 1. An array of byte arrays, containing the byte sequences of each entry.
        /// 2. An array of strings, containing the name of each entry.
        ///
        /// This method is used for comparing pipelines. Its outputs can be passed to <see cref="CheckSamePipeline"/>
        /// to check if this pipeline is identical to another pipeline.
        /// </summary>
        public static void SerializeRoleMappedData(IHostEnvironment env, IChannel ch, RoleMappedData data,
            out byte[][] dataSerialized, out string[] dataZipEntryNames)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ch, nameof(ch));
            ch.CheckValue(data, nameof(data));

            using (var ms = new MemoryStream())
            {
                TrainUtils.SaveModel(env, ch, ms, null, data);
                var zip = new ZipArchive(ms);
                var entries = zip.Entries.OrderBy(e => e.FullName).ToArray();
                dataSerialized = new byte[Utils.Size(entries)][];
                dataZipEntryNames = new string[Utils.Size(entries)];
                for (int i = 0; i < Utils.Size(entries); i++)
                {
                    dataZipEntryNames[i] = entries[i].FullName;
                    dataSerialized[i] = new byte[entries[i].Length];
                    using (var s = entries[i].Open())
                        s.Read(dataSerialized[i], 0, (int)entries[i].Length);
                }
            }
        }

        /// <summary>
        /// This method compares two pipelines to make sure they are identical. The first pipeline is passed
        /// as a <see cref="RoleMappedData"/>, and the second as a double byte array and a string array. The double
        /// byte array and the string array are obtained by calling <see cref="SerializeRoleMappedData"/> on the
        /// second pipeline.
        /// The comparison is done by saving <see ref="dataToCompare"/> as an in-memory <see cref="ZipArchive"/>,
        /// and for each entry in it, comparing its name, and the byte sequence to the corresponding entries in
        /// <see ref="dataZipEntryNames"/> and <see ref="dataSerialized"/>.
        /// This method throws if for any of the entries the name/byte sequence are not identical.
        /// </summary>
        public static void CheckSamePipeline(IHostEnvironment env, IChannel ch,
            RoleMappedData dataToCompare, byte[][] dataSerialized, string[] dataZipEntryNames)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ch, nameof(ch));
            ch.CheckValue(dataToCompare, nameof(dataToCompare));
            ch.CheckValue(dataSerialized, nameof(dataSerialized));
            ch.CheckValue(dataZipEntryNames, nameof(dataZipEntryNames));
            if (dataZipEntryNames.Length != dataSerialized.Length)
            {
                throw ch.ExceptParam(nameof(dataSerialized),
                    $"The length of {nameof(dataSerialized)} must be equal to the length of {nameof(dataZipEntryNames)}");
            }

            using (var ms = new MemoryStream())
            {
                // REVIEW: This can be done more efficiently by adding a custom type of repository that
                // doesn't actually save the data, but upon stream closure compares the results to the given repository
                // and then discards it. Currently, however, this cannot be done because ModelSaveContext does not use
                // an abstract class/interface, but rather the RepositoryWriter class.
                TrainUtils.SaveModel(env, ch, ms, null, dataToCompare);

                string errorMsg = "Models contain different pipelines, cannot ensemble them.";
                var zip = new ZipArchive(ms);
                var entries = zip.Entries.OrderBy(e => e.FullName).ToArray();
                ch.Check(dataSerialized.Length == Utils.Size(entries));
                byte[] buffer = null;
                for (int i = 0; i < dataSerialized.Length; i++)
                {
                    ch.Check(dataZipEntryNames[i] == entries[i].FullName, errorMsg);
                    int len = dataSerialized[i].Length;
                    if (Utils.Size(buffer) < len)
                        buffer = new byte[len];
                    using (var s = entries[i].Open())
                    {
                        int bytesRead = s.Read(buffer, 0, len);
                        ch.Check(bytesRead == len, errorMsg);
                        for (int j = 0; j < len; j++)
                            ch.Check(buffer[j] == dataSerialized[i][j], errorMsg);
                        if (s.Read(buffer, 0, 1) > 0)
                            throw env.Except(errorMsg);
                    }
                }
            }
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Model;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public sealed class LoaderWrapper : IDataReader<IMultiStreamSource>, ICanSaveModel
    {
        public const string LoaderSignature = "LoaderWrapper";

        private readonly IHostEnvironment _env;
        private readonly Func<IMultiStreamSource, IDataView> _loaderFactory;

        public LoaderWrapper(IHostEnvironment env, Func<IMultiStreamSource, IDataView> loaderFactory)
        {
            _env = env;
            _loaderFactory = loaderFactory;
        }

        public ISchema GetOutputSchema()
        {
            var emptyData = Read(new MultiFileSource(null));
            return emptyData.Schema;
        }

        public IDataView Read(IMultiStreamSource input) => _loaderFactory(input);

        public void Save(ModelSaveContext ctx)
        {
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());
            var ldr = Read(new MultiFileSource(null));
            ctx.SaveModel(ldr, "Loader");
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "LDR WRPR",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public LoaderWrapper(IHostEnvironment env, ModelLoadContext ctx)
        {
            ctx.CheckAtModel(GetVersionInfo());
            ctx.LoadModel<IDataLoader, SignatureLoadDataLoader>(env, out var loader, "Loader", new MultiFileSource(null));

            var loaderStream = new MemoryStream();
            using (var rep = RepositoryWriter.CreateNew(loaderStream))
            {
                ModelSaveContext.SaveModel(rep, loader, "Loader");
                rep.Commit();
            }

            _env = env;
            _loaderFactory = (IMultiStreamSource source) =>
            {
                using (var rep = RepositoryReader.Open(loaderStream))
                {
                    ModelLoadContext.LoadModel<IDataLoader, SignatureLoadDataLoader>(env, out var ldr, rep, "Loader", source);
                    return ldr;
                }
            };

        }
    }

    public sealed class MyBinaryClassifierEvaluator
    {
        private readonly IHostEnvironment _env;
        private readonly BinaryClassifierEvaluator _evaluator;

        public MyBinaryClassifierEvaluator(IHostEnvironment env, BinaryClassifierEvaluator.Arguments args)
        {
            _env = env;
            _evaluator = new BinaryClassifierEvaluator(env, args);
        }

        public BinaryClassificationMetrics Evaluate(IDataView data, string labelColumn = DefaultColumnNames.Label,
            string probabilityColumn = DefaultColumnNames.Probability)
        {
            var ci = EvaluateUtils.GetScoreColumnInfo(_env, data.Schema, null, DefaultColumnNames.Score, MetadataUtils.Const.ScoreColumnKind.BinaryClassification);
            var map = new KeyValuePair<RoleMappedSchema.ColumnRole, string>[]
            {
                RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Probability, probabilityColumn),
                RoleMappedSchema.CreatePair(MetadataUtils.Const.ScoreValueKind.Score, ci.Name)
            };
            var rmd = new RoleMappedData(data, labelColumn, DefaultColumnNames.Features, opt: true, custom: map);

            var metricsDict = _evaluator.Evaluate(rmd);
            return BinaryClassificationMetrics.FromMetrics(_env, metricsDict["OverallMetrics"], metricsDict["ConfusionMatrix"]).Single();
        }
    }

    public static class MyCrossValidation
    {
        public sealed class BinaryCrossValidationMetrics
        {
            public readonly ITransformer[] FoldModels;
            public readonly BinaryClassificationMetrics[] FoldMetrics;

            public BinaryCrossValidationMetrics(ITransformer[] models, BinaryClassificationMetrics[] metrics)
            {
                FoldModels = models;
                FoldMetrics = metrics;
            }
        }

        public sealed class BinaryCrossValidator
        {
            private readonly IHostEnvironment _env;

            public int NumFolds { get; set; } = 2;

            public string StratificationColumn { get; set; }

            public string LabelColumn { get; set; } = DefaultColumnNames.Label;

            public BinaryCrossValidator(IHostEnvironment env)
            {
                _env = env;
            }

            public BinaryCrossValidationMetrics CrossValidate(IDataView trainData, IEstimator<ITransformer> estimator)
            {
                var models = new ITransformer[NumFolds];
                var metrics = new BinaryClassificationMetrics[NumFolds];

                if (StratificationColumn == null)
                {
                    StratificationColumn = "StratificationColumn";
                    var random = new GenerateNumberTransform(_env, trainData, StratificationColumn);
                    trainData = random;
                }
                else
                    throw new NotImplementedException();

                var evaluator = new MyBinaryClassifierEvaluator(_env, new BinaryClassifierEvaluator.Arguments() { });

                for (int fold = 0; fold < NumFolds; fold++)
                {
                    var trainFilter = new RangeFilter(_env, new RangeFilter.Arguments()
                    {
                        Column = StratificationColumn,
                        Min = (Double)fold / NumFolds,
                        Max = (Double)(fold + 1) / NumFolds,
                        Complement = true
                    }, trainData);
                    var testFilter = new RangeFilter(_env, new RangeFilter.Arguments()
                    {
                        Column = StratificationColumn,
                        Min = (Double)fold / NumFolds,
                        Max = (Double)(fold + 1) / NumFolds,
                        Complement = false
                    }, trainData);

                    models[fold] = estimator.Fit(trainFilter);
                    var scoredTest = models[fold].Transform(testFilter);
                    metrics[fold] = evaluator.Evaluate(scoredTest, labelColumn: LabelColumn, probabilityColumn: "Probability");
                }

                return new BinaryCrossValidationMetrics(models, metrics);

            }
        }
    }

    public class MyLambdaTransform<TSrc, TDst> : IEstimator<TransformWrapper>
         where TSrc : class, new()
            where TDst : class, new()
    {
        private readonly IHostEnvironment _env;
        private readonly Action<TSrc, TDst> _action;

        public MyLambdaTransform(IHostEnvironment env, Action<TSrc, TDst> action)
        {
            _env = env;
            _action = action;
        }

        public TransformWrapper Fit(IDataView input)
        {
            var xf = LambdaTransform.CreateMap(_env, input, _action);
            var empty = new EmptyDataView(_env, input.Schema);
            var chunk = ApplyTransformUtils.ApplyAllTransformsToData(_env, xf, empty, input);
            return new TransformWrapper(_env, chunk);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }
    }

    public static class MyHelperExtensions
    {
        public static void SaveAsBinary(this IDataView data, IHostEnvironment env, Stream stream)
        {
            var saver = new BinarySaver(env, new BinarySaver.Arguments());
            using (var ch = env.Start("SaveData"))
                DataSaverUtils.SaveDataView(ch, saver, data, stream);
        }

        public static IDataView FitAndTransform(this IEstimator<ITransformer> est, IDataView data) => est.Fit(data).Transform(data);

        public static IDataView FitAndRead<TSource>(this IDataReaderEstimator<TSource, IDataReader<TSource>> est, TSource source)
            => est.Fit(source).Read(source);
    }
}

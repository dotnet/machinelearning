// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms.TimeSeries;

[assembly: LoadableClass(SrCnnAnomalyDetector.Summary, typeof(IDataTransform), typeof(SrCnnAnomalyDetector), typeof(SrCnnAnomalyDetector.Options), typeof(SignatureDataTransform),
    SrCnnAnomalyDetector.UserName, SrCnnAnomalyDetector.LoaderSignature, SrCnnAnomalyDetector.ShortName)]

[assembly: LoadableClass(SrCnnAnomalyDetector.Summary, typeof(IDataTransform), typeof(SrCnnAnomalyDetector), null, typeof(SignatureLoadDataTransform),
    SrCnnAnomalyDetector.UserName, SrCnnAnomalyDetector.LoaderSignature)]

[assembly: LoadableClass(SrCnnAnomalyDetector.Summary, typeof(SrCnnAnomalyDetector), null, typeof(SignatureLoadModel),
    SrCnnAnomalyDetector.UserName, SrCnnAnomalyDetector.LoaderSignature)]

[assembly: LoadableClass(typeof(IRowMapper), typeof(SrCnnAnomalyDetector), null, typeof(SignatureLoadRowMapper),
   SrCnnAnomalyDetector.UserName, SrCnnAnomalyDetector.LoaderSignature)]

namespace Microsoft.ML.Transforms.TimeSeries
{
    public sealed class SrCnnAnomalyDetector : SrCnnAnomalyDetectionBaseWrapper, IStatefulTransformer
    {
        internal const string Summary = "This transform detects the anomalies in a time-series using SRCNN.";
        internal const string LoaderSignature = "SrCnnAnomalyDetector";
        internal const string UserName = "SrCnn Anomaly Detection";
        internal const string ShortName = "srcnn";

        internal sealed class Options : TransformInputBase
        {
            [Argument(ArgumentType.Required, HelpText = "The name of the source column.", ShortName = "src",
                SortOrder = 1, Purpose = SpecialPurpose.ColumnName)]
            public string Source;

            [Argument(ArgumentType.Required, HelpText = "The name of the new column.",
                SortOrder = 2)]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "The size of the sliding window for computing spectral residual", ShortName = "wnd",
                SortOrder = 101)]
            public int WindowSize = 24;

            [Argument(ArgumentType.Required, HelpText = "The number of points to the back of training window.",
                ShortName = "bwnd", SortOrder = 102)]
            public int BackAddWindowSize = 5;

            [Argument(ArgumentType.Required, HelpText = "The number of pervious points used in prediction.",
                ShortName = "awnd", SortOrder = 103)]
            public int LookaheadWindowSize = 5;

            [Argument(ArgumentType.Required, HelpText = "The threshold to determine anomaly, score larger than the threshold is considered as anomaly.",
                ShortName = "thre", SortOrder = 104)]
            public double Threshold = 0.3;
        }

        private sealed class SrCnnArgument : SrCnnArgumentBase
        {
            public SrCnnArgument(Options options)
            {
                Source = options.Source;
                Name = options.Name;
                WindowSize = options.WindowSize;
                InitialWindowSize = 0;
                BackAddWindowSize = options.BackAddWindowSize;
                LookaheadWindowSize = options.LookaheadWindowSize;
                Threshold = options.Threshold;
            }

            public SrCnnArgument(SrCnnAnomalyDetector transform)
            {
                Source = transform.InternalTransform.InputColumnName;
                Name = transform.InternalTransform.OutputColumnName;
                WindowSize = transform.InternalTransform.WindowSize;
                InitialWindowSize = 0;
                BackAddWindowSize = transform.InternalTransform.BackAddWindowSize;
                LookaheadWindowSize = transform.InternalTransform.LookaheadWindowSize;
                Threshold = transform.InternalTransform.AlertThreshold;
            }
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SRCNNTRNS",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(SrCnnAnomalyDetector).Assembly.FullName);
        }

        private static IDataTransform Create(IHostEnvironment env, Options options, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(options, nameof(options));
            env.CheckValue(input, nameof(input));

            return new SrCnnAnomalyDetector(env, options).MakeDataTransform(input);
        }

        private static IDataTransform Create(IHostEnvironment env, ModelLoadContext ctx, IDataView input)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            env.CheckValue(input, nameof(input));

            return new SrCnnAnomalyDetector(env, ctx).MakeDataTransform(input);
        }

        private static SrCnnAnomalyDetector Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return new SrCnnAnomalyDetector(env, ctx);
        }

        private static IRowMapper Create(IHostEnvironment env, ModelLoadContext ctx, DataViewSchema inputSchema)
            => Create(env, ctx).MakeRowMapper(inputSchema);

        IStatefulTransformer IStatefulTransformer.Clone()
        {
            var clone = (SrCnnAnomalyDetector)MemberwiseClone();
            clone.InternalTransform.StateRef = (SrCnnAnomalyDetectionBase.State)clone.InternalTransform.StateRef.Clone();
            clone.InternalTransform.StateRef.InitState(clone.InternalTransform, InternalTransform.Host);
            return clone;
        }

        internal SrCnnAnomalyDetector(IHostEnvironment env, Options options)
            :base(new SrCnnArgument(options), LoaderSignature, env)
        {
        }

        internal SrCnnAnomalyDetector(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, LoaderSignature)
        {
            //TODO:
        }

        private SrCnnAnomalyDetector(IHostEnvironment env, SrCnnAnomalyDetector transform)
           : base(new SrCnnArgument(transform), LoaderSignature, env)
        {
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            //TODO:
        }
    }

    /// <summary>
    /// Detect anomalies in time series using Spectral Residual
    /// </summary>
    public sealed class SrCnnAnomalyEstimator : TrivialEstimator<SrCnnAnomalyDetector>
    {
        /// <summary>
        /// Create a new instance of <see cref="SrCnnAnomalyDetector"/>
        /// </summary>
        /// <param name="env"></param>
        /// <param name="outputColumnName"></param>
        /// <param name="windowSize"></param>
        /// <param name="backAddWindowSize"></param>
        /// <param name="lookaheadWindowSize"></param>
        /// <param name="threshold"></param>
        /// <param name="inputColumnName"></param>
        internal SrCnnAnomalyEstimator(IHostEnvironment env,
            string outputColumnName,
            int windowSize,
            int backAddWindowSize,
            int lookaheadWindowSize,
            double threshold = 0.3,
            string inputColumnName = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(SrCnnAnomalyEstimator)),
                  new SrCnnAnomalyDetector(env, new SrCnnAnomalyDetector.Options
                  {
                      Source = inputColumnName ?? outputColumnName,
                      Name = outputColumnName,
                      WindowSize = windowSize,
                      BackAddWindowSize = backAddWindowSize,
                      LookaheadWindowSize = lookaheadWindowSize,
                      Threshold = threshold
                  }))
        {
        }

        internal SrCnnAnomalyEstimator(IHostEnvironment env, SrCnnAnomalyDetector.Options options)
            : base(Contracts.CheckRef(env, nameof(env)).Register(nameof(SrCnnAnomalyEstimator)), new SrCnnAnomalyDetector(env, options))
        {
        }

        public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            //TODO:
            throw new NotImplementedException();
        }

    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
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
    /// <summary>
    /// <see cref="ITransformer"/> resulting from fitting a <see cref="SrCnnAnomalyEstimator"/>.
    /// </summary>
    public sealed class SrCnnAnomalyDetector : SrCnnAnomalyDetectionBase, IStatefulTransformer
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
                ShortName = "backwnd", SortOrder = 102)]
            public int BackAddWindowSize = 5;

            [Argument(ArgumentType.Required, HelpText = "The number of pervious points used in prediction.",
                ShortName = "aheadwnd", SortOrder = 103)]
            public int LookaheadWindowSize = 5;

            [Argument(ArgumentType.Required, HelpText = "The size of sliding window to generate a saliency map for the series.",
                ShortName = "avgwnd", SortOrder = 104)]
            public int AvergingWindowSize = 3;

            [Argument(ArgumentType.Required, HelpText = "The size of sliding window to calculate the anomaly score for each data point.",
                ShortName = "jdgwnd", SortOrder = 105)]
            public int JudgementWindowSize = 21;

            [Argument(ArgumentType.Required, HelpText = "The threshold to determine anomaly, score larger than the threshold is considered as anomaly.",
                ShortName = "thre", SortOrder = 106)]
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
                AvergingWindowSize = options.AvergingWindowSize;
                JudgementWindowSize = options.JudgementWindowSize;
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
                AvergingWindowSize = transform.InternalTransform.AvergingWindowSize;
                JudgementWindowSize = transform.InternalTransform.JudgementWindowSize;
                Threshold = transform.InternalTransform.AlertThreshold;
            }
        }

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "SRCNTRNS",
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

        internal static SrCnnAnomalyDetector Create(IHostEnvironment env, ModelLoadContext ctx)
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
            clone.InternalTransform.StateRef = (SrCnnAnomalyDetectionBaseCore.State)clone.InternalTransform.StateRef.Clone();
            clone.InternalTransform.StateRef.InitState(clone.InternalTransform, InternalTransform.Host);
            return clone;
        }

        internal SrCnnAnomalyDetector(IHostEnvironment env, Options options)
            : base(new SrCnnArgument(options), LoaderSignature, env)
        {
        }

        private SrCnnAnomalyDetector(IHostEnvironment env, ModelLoadContext ctx)
            : base(env, ctx, LoaderSignature)
        {
        }

        private SrCnnAnomalyDetector(IHostEnvironment env, SrCnnAnomalyDetector transform)
           : base(new SrCnnArgument(transform), LoaderSignature, env)
        {
        }

        private protected override void SaveModel(ModelSaveContext ctx)
        {
            InternalTransform.Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // <base>
            base.SaveModel(ctx);
        }
    }

    /// <summary>
    /// Detect anomalies in time series using Spectral Residual(SR) algorithm
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this estimator, use
    /// [DetectAnomalyBySrCnn](xref:Microsoft.ML.TimeSeriesCatalog.DetectAnomalyBySrCnn(Microsoft.ML.TransformsCatalog,System.String,System.String,System.Int32,System.Int32,System.Int32,System.Int32,System.Int32,System.Double))
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | No |
    /// | Input column data type | <xref:System.Single> |
    /// | Output column data type | 3-element vector of<xref:System.Double> |
    /// | Exportable to ONNX | No |
    ///
    /// ### Background
    /// At Microsoft, we have developed a time-series anomaly detection service which helps customers to monitor the time-series continuously
    /// and alert for potential incidents on time. To tackle the problem of time-series anomaly detection,
    /// we propose a novel algorithm based on Spectral Residual (SR) and Convolutional Neural Network
    /// (CNN). The SR model is borrowed from visual saliency detection domain to time-series anomaly detection.
    /// And here we onboarded this SR algorithm firstly.
    ///
    /// The Spectral Residual (SR) algorithm is unsupervised, which means training step is not needed while using SR. It consists of three major steps:
    /// (1) Fourier Transform to get the log amplitude spectrum;
    /// (2) calculation of spectral residual;
    /// (3) Inverse Fourier Transform that transforms the sequence back to spatial domain.
    /// Mathematically, given a sequence $\mathbf{x}$, we have
    /// $$A(f) = Amplitude(\mathfrak{F}(\mathbf{x}))\\P(f) = Phrase(\mathfrak{F}(\mathbf{x}))\\L(f) = log(A(f))\\AL(f) = h_n(f) \cdot L(f)\\R(f) = L(f) - AL(f)\\S(\mathbf{x}) = \mathfrak{F}^{-1}(exp(R(f) + P(f))^{2})$$
    /// where $\mathfrak{F}$ and $\mathfrak{F}^{-1}$ denote Fourier Transform and Inverse Fourier Transform respectively.
    /// $\mathbf{x}$ is the input sequence with shape $n × 1$; $A(f)$ is the amplitude spectrum of sequence $\mathbf{x}$;
    /// $P(f)$ is the corresponding phase spectrum of sequence $\mathbf{x}$; $L(f)$ is the log representation of $A(f)$;
    /// and $AL(f)$ is the average spectrum of $L(f)$ which can be approximated by convoluting the input sequence by $h_n(f)$,
    /// where $h_n(f)$ is an $n × n$ matrix defined as:
    /// $$n_f(f) = \begin{bmatrix}1&1&1&\cdots&1\\1&1&1&\cdots&1\\\vdots&\vdots&\vdots&\ddots&\vdots\\1&1&1&\cdots&1\end{bmatrix}$$
    /// $R(f)$ is the spectral residual, i.e., the log spectrum $L(f)$ subtracting the averaged log spectrum $AL(f)$.
    /// The spectral residual serves as a compressed representation of the sequence while the innovation part of the original sequence becomes more significant.
    /// At last, we transfer the sequence back to spatial domain via Inverse Fourier Transform. The result sequence $S(\mathbf{x})$ is called the saliency map.
    /// Given the saliency map $S(\mathbf{x})$, the output sequence $O(\mathbf{x})$ is computed by:
    /// $$O(x_i) = \begin{cases}1, if \frac{S(x_i)-\overline{S(x_i)}}{S(x_i)} > \tau\\0,otherwise,\end{cases}$$
    /// where $x_i$ represents an arbitrary point in sequence $\mathbf{x}$; $S(x_i)$is the corresponding point in the saliency map;
    /// and $\overline{S(x_i)}$ is the local average of the preceding points of $S(x_i)$.
    ///
    /// There are several parameters for SR algorithm. To obtain a model with good performance,
    /// we suggest to tune <strong>windowSize</strong> and <strong>threshold</strong> at first,
    /// these are the most important parameters to SR. Then you could search for an appropriate <strong>judgementWindowSize</strong>
    /// which is no larger than <strong>windowSize</strong>. And for the remaining parameters, you could use the default value directly.
    ///
    /// For more details please refer to the <a href="https://arxiv.org/pdf/1906.03821">Time-Series Anomaly Detection Service at Microsoft</a> paper.
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="Microsoft.ML.TimeSeriesCatalog.DetectAnomalyBySrCnn(TransformsCatalog, string, string, int, int, int, int, int, double)"/>
    public sealed class SrCnnAnomalyEstimator : TrivialEstimator<SrCnnAnomalyDetector>
    {
        /// <param name="env">Host environment.</param>
        /// <param name="outputColumnName">Name of the column resulting from the transformation of <paramref name="inputColumnName"/>.</param>
        /// <param name="windowSize">The size of the sliding window for computing spectral residual.</param>
        /// <param name="backAddWindowSize">The size of the sliding window for computing spectral residual.</param>
        /// <param name="lookaheadWindowSize">The number of pervious points used in prediction.</param>
        /// <param name="averagingWindowSize">The size of sliding window to generate a saliency map for the series.</param>
        /// <param name="judgementWindowSize">The size of sliding window to calculate the anomaly score for each data point.</param>
        /// <param name="threshold">The threshold to determine anomaly, score larger than the threshold is considered as anomaly.</param>
        /// <param name="inputColumnName">Name of column to transform. The column data must be <see cref="System.Single"/>.</param>
        internal SrCnnAnomalyEstimator(IHostEnvironment env,
            string outputColumnName,
            int windowSize,
            int backAddWindowSize,
            int lookaheadWindowSize,
            int averagingWindowSize,
            int judgementWindowSize,
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
                      AvergingWindowSize = averagingWindowSize,
                      JudgementWindowSize = judgementWindowSize,
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
            Host.CheckValue(inputSchema, nameof(inputSchema));

            if (!inputSchema.TryFindColumn(Transformer.InternalTransform.InputColumnName, out var col))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", Transformer.InternalTransform.InputColumnName);
            if (col.ItemType != NumberDataViewType.Single)
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "input", Transformer.InternalTransform.InputColumnName, "Single", col.GetTypeString());

            var metadata = new List<SchemaShape.Column>() {
                new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextDataViewType.Instance, false)
            };
            var resultDic = inputSchema.ToDictionary(x => x.Name);
            resultDic[Transformer.InternalTransform.OutputColumnName] = new SchemaShape.Column(
                Transformer.InternalTransform.OutputColumnName, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Double, false, new SchemaShape(metadata));

            return new SchemaShape(resultDic.Values);
        }

    }
}

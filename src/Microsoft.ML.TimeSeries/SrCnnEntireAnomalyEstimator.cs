// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms.TimeSeries
{
    /// <summary>
    /// Detect timeseries anomalies for entire input using Spectral Residual(SR) algorithm.
    /// </summary>
    /// <remarks>
    /// <format type="text/markdown"><![CDATA[
    /// To create this estimator, use
    /// [DetectEntireAnomalyBySrCnn](xref:Microsoft.ML.TimeSeriesCatalog.DetectEntireAnomalyBySrCnn(Microsoft.ML.TransformsCatalog,System.String,System.String,System.Double,System.Int32,SrCnnDetectMode,System.Double))
    /// ###  Estimator Characteristics
    /// |  |  |
    /// | -- | -- |
    /// | Does this estimator need to look at the data to train its parameters? | Yes |
    /// | Input column data type | <xref:System.Single> |
    /// | Output column data type | vector of<xref:System.Double> |
    /// | Exportable to ONNX | No |
    ///
    /// ### Background
    /// In Microsoft, we developed a time-series anomaly detection service which helps customers to monitor the time-series continuously
    /// and alert for potential incidents on time. To tackle the problem of time-series anomaly detection,
    /// we proposed a novel algorithm based on Spectral Residual (SR) and Convolutional Neural Network
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
    /// * [Link to the KDD 2019 paper](https://dl.acm.org/doi/10.1145/3292500.3330680)
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="Microsoft.ML.TimeSeriesCatalog.DetectEntireAnomalyBySrCnn(TransformsCatalog, string, string, string, double, int, SrCnnDetectMode, double)"/>
    public sealed class SrCnnEntireAnomalyEstimator : IEstimator<SrCnnEntireTransformer>
    {
        private readonly IHost _host;
        private readonly SrCnnEntireTransformer.Options _options;
        private readonly string _timestampColumnName;
        private readonly string _valueColumnName;
        private readonly string _outputColumnName;

        internal SrCnnEntireAnomalyEstimator(IHostEnvironment env,
            string outputColumnName,
            string timestampColumnName = null,
            string valueColumnName = null,
            double threshold = 0.3,
            int batchSize = 1024,
            SrCnnDetectMode srCnnDetectMode = SrCnnDetectMode.AnomalyOnly,
            double sensitivity = 99.0)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(SrCnnEntireAnomalyEstimator));
            _timestampColumnName = timestampColumnName ?? outputColumnName;
            _valueColumnName = valueColumnName ?? outputColumnName;
            _outputColumnName = outputColumnName;
            _options = new SrCnnEntireTransformer.Options
            {
                Source = new SrCnnEntireTransformer.Column { Name="TsPoint", Source = new string[] { _timestampColumnName, _valueColumnName } },
                Target = _outputColumnName,
                Threshold = threshold,
                BatchSize = batchSize,
                SrCnnDetectMode = srCnnDetectMode,
                Sensitivity = sensitivity
            };
        }

        public SrCnnEntireTransformer Fit(IDataView input) => new SrCnnEntireTransformer(_host, _options, input);

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            _host.CheckValue(inputSchema, nameof(inputSchema));

            if (!inputSchema.TryFindColumn(_timestampColumnName, out var timestampCol))
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "inputTimestamp", _timestampColumnName);
            if (!(timestampCol.ItemType is DateTimeDataViewType))
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "inputTimestamp", _timestampColumnName, "DateTime", timestampCol.GetTypeString());

            if (!inputSchema.TryFindColumn(_valueColumnName, out var valueCol))
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "inputValue", _valueColumnName);
            if (valueCol.ItemType != NumberDataViewType.Double)
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "inputValue", _valueColumnName, "Double", valueCol.GetTypeString());

            var metadata = new List<SchemaShape.Column>() {
                new SchemaShape.Column(AnnotationUtils.Kinds.SlotNames, SchemaShape.Column.VectorKind.Vector, TextDataViewType.Instance, false)
            };
            var resultDic = inputSchema.ToDictionary(x => x.Name);
            resultDic[_outputColumnName] = new SchemaShape.Column(
                _outputColumnName, SchemaShape.Column.VectorKind.Vector, NumberDataViewType.Double, false, new SchemaShape(metadata));

            return new SchemaShape(resultDic.Values);
        }
    }
}

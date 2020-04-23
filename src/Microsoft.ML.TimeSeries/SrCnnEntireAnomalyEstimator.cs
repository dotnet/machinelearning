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
    /// Please refer to <see cref="SrCnnAnomalyEstimator"/> for background of SR algorithm.
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
    /// ]]>
    /// </format>
    /// </remarks>
    /// <seealso cref="Microsoft.ML.TimeSeriesCatalog.DetectEntireAnomalyBySrCnn(TransformsCatalog, string, string, double, int, SrCnnDetectMode, double)"/>
    public sealed class SrCnnEntireAnomalyEstimator : IEstimator<SrCnnEntireTransformer>
    {
        private readonly IHost _host;
        private readonly SrCnnEntireTransformer.Options _options;
        private readonly string _inputColumnName;
        private readonly string _outputColumnName;

        internal SrCnnEntireAnomalyEstimator(IHostEnvironment env,
            string outputColumnName,
            string inputColumnName = null,
            double threshold = 0.3,
            int batchSize = 1024,
            SrCnnDetectMode srCnnDetectMode = SrCnnDetectMode.AnomalyOnly,
            double sensitivity = 99.0)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(nameof(SrCnnEntireAnomalyEstimator));
            _inputColumnName = inputColumnName ?? outputColumnName;
            _outputColumnName = outputColumnName;
            _options = new SrCnnEntireTransformer.Options
            {
                Source = _inputColumnName,
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

            if (!inputSchema.TryFindColumn(_inputColumnName, out var col))
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", _inputColumnName);
            if (!(col.ItemType is SrCnnTsPointDataViewType))
                throw _host.ExceptSchemaMismatch(nameof(inputSchema), "input", _inputColumnName, "SrCnnTsPointDataViewType", col.GetTypeString());

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

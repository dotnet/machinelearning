// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

namespace Microsoft.ML
{
    public class PredictionModel
    {
        private readonly Runtime.EntryPoints.TransformModel _predictorModel;
        private readonly IHostEnvironment _env;

        internal PredictionModel(Stream stream)
        {
            _env = new TlcEnvironment();
            _predictorModel = new Runtime.EntryPoints.TransformModel(_env, stream);
        }

        internal Runtime.EntryPoints.TransformModel PredictorModel
        {
            get { return _predictorModel; }
        }

        /// <summary>
        /// Returns labels that correspond to indices of the score array in the case of 
        /// multi-class classification problem.
        /// </summary>
        /// <param name="names">Label to score mapping</param>
        /// <param name="scoreColumnName">Name of the score column</param>
        /// <returns></returns>
        public bool TryGetScoreLabelNames(out string[] names, string scoreColumnName = DefaultColumnNames.Score)
        {
            names = null;
            ISchema schema = _predictorModel.OutputSchema;
            int colIndex = -1;
            if (!schema.TryGetColumnIndex(scoreColumnName, out colIndex))
                return false;
            
            int expectedLabelCount = schema.GetColumnType(colIndex).ValueCount;
            if (!schema.HasSlotNames(colIndex, expectedLabelCount))
                return false;

            VBuffer<DvText> labels = default;
            schema.GetMetadata(MetadataUtils.Kinds.SlotNames, colIndex, ref labels);

            if (labels.Length != expectedLabelCount)
                return false;

            names = new string[expectedLabelCount];
            int index = 0;
            foreach(var label in labels.DenseValues())
                names[index++] = label.ToString();

            return true;
        }

        /// <summary>
        /// Read model from file asynchronously.
        /// </summary>
        /// <param name="path">Path to the file</param>
        /// <returns>Model</returns>
        public static Task<PredictionModel> ReadAsync(string path)
        {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentNullException(nameof(path));

            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                return ReadAsync(stream);
            }
        }

        /// <summary>
        /// Read model from stream asynchronously.
        /// </summary>
        /// <param name="stream">Stream with model</param>
        /// <returns>Model</returns>
        public static Task<PredictionModel> ReadAsync(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            return Task.FromResult(new PredictionModel(stream));
        }

        /// <summary>
        /// Read generic model from file.
        /// </summary>
        /// <typeparam name="TInput">Type for incoming data</typeparam>
        /// <typeparam name="TOutput">Type for output data</typeparam>
        /// <param name="path">Path to the file</param>
        /// <returns>Model</returns>
        public static Task<PredictionModel<TInput, TOutput>> ReadAsync<TInput, TOutput>(string path)
            where TInput : class
            where TOutput : class, new()
        {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentNullException(nameof(path));

            using (var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                return ReadAsync<TInput, TOutput>(stream);
            }
        }

        /// <summary>
        /// Read generic model from file.
        /// </summary>
        /// <typeparam name="TInput">Type for incoming data</typeparam>
        /// <typeparam name="TOutput">Type for output data</typeparam>
        /// <param name="stream">Stream with model</param>
        /// <returns>Model</returns>
        public static Task<PredictionModel<TInput, TOutput>> ReadAsync<TInput, TOutput>(Stream stream)
            where TInput : class
            where TOutput : class, new()
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));

            using (var environment = new TlcEnvironment())
            {
                BatchPredictionEngine<TInput, TOutput> predictor =
                    environment.CreateBatchPredictionEngine<TInput, TOutput>(stream);

                return Task.FromResult(new PredictionModel<TInput, TOutput>(predictor, stream));
            }
        }

        /// <summary>
        /// Run prediction on top of IDataView.
        /// </summary>
        /// <param name="input">Incoming IDataView</param>
        /// <returns>IDataView which contains predictions</returns>
        public IDataView Predict(IDataView input) => _predictorModel.Apply(_env, input);

        /// <summary>
        /// Save model to file.
        /// </summary>
        /// <param name="path">File to save model</param>
        /// <returns></returns>
        public Task WriteAsync(string path)
        {
            if (string.IsNullOrEmpty(path))
                throw new ArgumentNullException(nameof(path));

            using (var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read))
            {
                return WriteAsync(stream);
            }
        }

        /// <summary>
        /// Save model to stream.
        /// </summary>
        /// <param name="stream">Stream to save model.</param>
        /// <returns></returns>
        public Task WriteAsync(Stream stream)
        {
            if (stream == null)
                throw new ArgumentNullException(nameof(stream));
            _predictorModel.Save(_env, stream);
            return Task.CompletedTask;
        }
    }

    public class PredictionModel<TInput, TOutput> : PredictionModel
        where TInput : class
        where TOutput : class, new()
    {
        private BatchPredictionEngine<TInput, TOutput> _predictor;

        internal PredictionModel(BatchPredictionEngine<TInput, TOutput> predictor, Stream stream)
            : base(stream)
        {
            _predictor = predictor;
        }

        /// <summary>
        /// Run prediction for the TInput data.
        /// </summary>
        /// <param name="input">Input data</param>
        /// <returns>Result of prediction</returns>
        public TOutput Predict(TInput input)
        {
            int count = 0;
            TOutput result = null;
            foreach (var item in _predictor.Predict(new[] { input }, reuseRowObjects: false))
            {
                if (count == 0)
                    result = item;

                count++;
                if (count > 1)
                    break;
            }

            if (count > 1)
                throw new InvalidOperationException("Prediction pipeline must return at most one prediction per example.");
            return result;
        }

        /// <summary>
        /// Run prediction for collection of inputs.
        /// </summary>
        /// <param name="inputs">Input data</param>
        /// <returns>Result of prediction</returns>
        public IEnumerable<TOutput> Predict(IEnumerable<TInput> inputs)
        {
            return _predictor.Predict(inputs, reuseRowObjects: false);
        }
    }
}
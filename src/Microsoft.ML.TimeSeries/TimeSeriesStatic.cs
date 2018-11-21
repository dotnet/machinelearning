// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.TimeSeriesProcessing;
using Microsoft.ML.StaticPipe;
using Microsoft.ML.StaticPipe.Runtime;
using System;
using System.Collections.Generic;
using static Microsoft.ML.Runtime.TimeSeriesProcessing.SequentialAnomalyDetectionTransformBase<System.Single, Microsoft.ML.Runtime.TimeSeriesProcessing.IidAnomalyDetectionBase.State>;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// IidChangePoint static API extension methods.
    /// </summary>
    public static class IidChangePointStaticExtensions
    {
        public class Prediction
        {
            public int Alert;
            public float Score;
            public float PValue;
            public float MartingaleValue;
        }

        private sealed class OutColumn : Vector<Prediction>
        {
            public PipelineColumn Input { get; }

            public OutColumn(
                Vector<float> input,
                int confidence,
                int changeHistoryLength,
                MartingaleType martingale,
                double eps)
                : base(new Reconciler(confidence, changeHistoryLength, martingale, eps), input)
            {
                Input = input;
            }
        }

        private sealed class Reconciler : EstimatorReconciler
        {
            private readonly int _confidence;
            private readonly int _changeHistoryLength;
            private readonly MartingaleType _martingale;
            private readonly double _eps;

            public Reconciler(
                int confidence,
                int changeHistoryLength,
                MartingaleType martingale,
                double eps)
            {
                _confidence = confidence;
                _changeHistoryLength = changeHistoryLength;
                _martingale = martingale;
                _eps = eps;
            }

            public override IEstimator<ITransformer> Reconcile(IHostEnvironment env,
                PipelineColumn[] toOutput,
                IReadOnlyDictionary<PipelineColumn, string> inputNames,
                IReadOnlyDictionary<PipelineColumn, string> outputNames,
                IReadOnlyCollection<string> usedNames)
            {
                Contracts.Assert(toOutput.Length == 1);
                var outCol = (OutColumn)toOutput[0];
                return new IidChangePointEstimator(env,
                    inputNames[outCol.Input],
                    outputNames[outCol],
                    _confidence,
                    _changeHistoryLength,
                    _martingale,
                    _eps
                    );
            }
        }

        public static Vector<Prediction> IidChangePointDetect(
            this Vector<float> input,
            int confidence,
            int changeHistoryLength,
            MartingaleType martingale = MartingaleType.Power,
            double eps = 0.1) => new OutColumn(input, confidence, changeHistoryLength, martingale, eps);

    }
}

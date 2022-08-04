// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Data.Analysis;

namespace Microsoft.ML.Fairlearn.reductions
{
    /// <summary>
    /// Generic moment.
    /// Modeled after the original Fairlearn <see href="https://github.com/fairlearn/fairlearn/blob/931963c40c0ba0cdd1a9e51c29adcc509da224a6/fairlearn/reductions/_moments/moment.py#L15">repo</see>
    /// Our implementations of the reductions approach to fairness
    /// <see href="https://arxiv.org/abs/1803.02453">agarwal2018reductions</see> 
    /// make use of Moment objects to describe both the optimization objective
    /// and the fairness constraints imposed on the solution.
    /// This is an abstract class for all such objects.
    /// </summary>
    public abstract class Moment
    {
        protected DataFrameColumn Y; //maybe lowercase this?
        public DataFrame Tags { get; private set; }
        public IDataView X { get; protected set; }
        public long TotalSamples { get; protected set; }

        public DataFrameColumn SensitiveFeatureColumn { get => Tags["group_id"]; }

        public Moment()
        {

        }
        /// <summary>
        /// Load the data into the moment to generate parity constarint
        /// </summary>
        /// <param name="x">The feature set</param>
        /// <param name="y">The label</param>
        /// <param name="sensitiveFeature">The sentivite featue that contain the sensitive groups</param>
        public void LoadData(IDataView x, DataFrameColumn y, StringDataFrameColumn sensitiveFeature)
        {

            X = x;
            TotalSamples = y.Length;
            Y = y;
            Tags = new DataFrame();
            Tags["label"] = y;

            Tags["group_id"] = sensitiveFeature;
        }

        /// <summary>
        /// Calculate the degree to which constraints are currently violated by the predictor.
        /// </summary>
        /// <param name="yPred">Contains the predictions of the label</param>
        /// <returns></returns>
        public abstract DataFrame Gamma(PrimitiveDataFrameColumn<float> yPred);
        public abstract float Bound();
        public float ProjectLambda()
        {
            throw new NotImplementedException();
        }
        public abstract DataFrameColumn SignedWeights(DataFrame lambdaVec);
    }
    /// <summary>
    /// Moment that can be expressed as weighted classification error.
    /// </summary>
    public abstract class ClassificationMoment : Moment
    {

    }
}

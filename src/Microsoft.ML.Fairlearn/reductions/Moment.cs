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
    /// General Moment of :class:`Moment` objects to describe the disparity constraints imposed
    /// on the solution.This is an abstract class for all such objects.
    /// </summary>
    public class Moment
    {
        private bool _dataLoaded = false;
        protected IDataView X; //uppercase?
        protected DataFrameColumn Y;
        protected DataFrame Tags;

        public Moment()
        {

        }
        public void LoadData(IDataView x, DataFrameColumn y, StringDataFrameColumn sensitiveFeature = null)
        {
            if (_dataLoaded)
            {
                throw new InvalidOperationException("data can be loaded only once");
            }

            X = x;
            Y = y;
            Tags = new DataFrame();
            Tags["label"] = y;

            if (sensitiveFeature != null)
            {
                // _tags["group_id"] = DataFrameColumn.Create; maybe convert from a vector?
                Tags["group_id"] = sensitiveFeature;
            }
            _dataLoaded = true;
        }

        public DataFrame Gamma(PrimitiveDataFrameColumn<float> yPred)
        {
            throw new NotImplementedException();
        }
        public float Bound()
        {
            throw new NotImplementedException();
        }
        public float ProjectLambda()
        {
            throw new NotImplementedException();
        }
        public float SignedWeights()
        {
            throw new NotImplementedException();
        }
    }
    /// <summary>
    /// Moment that can be expressed as weighted classification error.
    /// </summary>
    public class ClassificationMoment : Moment
    {

    }
}

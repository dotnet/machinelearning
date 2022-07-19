// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.Data.Analysis;
using Microsoft.ML.Data;

namespace Microsoft.ML.Fairlearn.reductions
{
    /// <summary>
    /// Grid Search. Right now only supports binary classification
    /// </summary>
    public class GridSearch
    {
        private readonly Moment _constraints;
        public GridSearch(Moment constraints, float constraintWeight = 0.5F, float gridSize = 10F, float gridLimit = 2.0F, float? gridOffset = null)
        {
            _constraints = constraints;
        }

        public void Fit(IDataView x, DataFrameColumn y, StringDataFrameColumn sensitiveFeature)
        {
            _constraints.LoadData(x, y, sensitiveFeature);

        }

    }
}


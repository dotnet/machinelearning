// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.Data.Analysis;

namespace Microsoft.ML.Fairlearn
{
    internal interface IGroupMetric
    {
        /// <summary>
        /// calculate metric all over group. It returns a dictionary where key is metric name
        /// and value is metric value
        /// </summary>
        Dictionary<string, double> Overall();

        /// <summary>
        /// calculate metric according to group. It returns a dataframe
        /// which index is each value in a group and column is metric name and metric name.
        /// </summary>
        DataFrame ByGroup();
    }
}

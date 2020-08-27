// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML
{
    internal interface IMetricsAgent<T>
    {
        double GetScore(T metrics);

        bool IsModelPerfect(double score);

        // GroupId is a parameter used only in RankingMetricsAgent
        T EvaluateMetrics(IDataView data, string labelColumn, string groupId);
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Legacy.Trainers
{
    /// <summary>
    /// This API requires Microsoft.ML.LightGBM nuget.
    /// </summary>
    /// <example>
    /// <code>
    /// pipeline.Add(new LightGbmBinaryClassifier() { NumLeaves = 5, NumBoostRound = 5, MinDataPerLeaf = 2 })
    /// </code>
    /// </example>
    public sealed partial class LightGbmBinaryClassifier
    {

    }

    /// <summary>
    /// This API requires Microsoft.ML.LightGBM nuget.
    /// </summary>
    /// <example>
    /// <code>
    /// pipeline.Add(new LightGbmClassifier() { NumLeaves = 5, NumBoostRound = 5, MinDataPerLeaf = 2 })
    /// </code>
    /// </example>
    public sealed partial class LightGbmClassifier
    {

    }

    /// <summary>
    /// This API requires Microsoft.ML.LightGBM nuget.
    /// </summary>
    /// <example>
    /// <code>
    /// pipeline.Add(new LightGbmRanker() { NumLeaves = 5, NumBoostRound = 5, MinDataPerLeaf = 2 })
    /// </code>
    /// </example>
    public sealed partial class LightGbmRanker
    {

    }

    /// <summary>
    /// This API requires Microsoft.ML.LightGBM nuget.
    /// </summary>
    /// <example>
    /// <code>
    /// pipeline.Add(new LightGbmRegressor() { NumLeaves = 5, NumBoostRound = 5, MinDataPerLeaf = 2 })
    /// </code>
    /// </example>
    public sealed partial class LightGbmRegressor
    {

    }
}

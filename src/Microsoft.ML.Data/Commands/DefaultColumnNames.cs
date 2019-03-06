// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Data
{
    /// <summary>
    /// A set of string literals intended to be "canonical" names for column names intended for particular purpose.
    /// While not part of the public API surface, its primary purpose is intended to be used in such a way as to encourage
    /// uniformity on the public API surface, wherever it is judged where columns with default names should be consumed.
    /// </summary>
    [BestFriend]
    internal static class DefaultColumnNames
    {
        public const string Features = "Features";
        public const string Label = "Label";
        public const string GroupId = "GroupId";
        public const string Name = "Name";
        public const string Weight = "Weight";
        public const string Score = "Score";
        public const string Probability = "Probability";
        public const string PredictedLabel = "PredictedLabel";
        public const string RecommendedItems = "Recommended";
        public const string User = "User";
        public const string Item = "Item";
        public const string Date = "Date";
        public const string FeatureContributions = "FeatureContributions";
    }
}

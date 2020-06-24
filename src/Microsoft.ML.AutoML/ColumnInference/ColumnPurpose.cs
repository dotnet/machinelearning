// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.AutoML
{
    internal enum ColumnPurpose
    {
        Ignore = 0,
        Label = 1,
        NumericFeature = 2,
        CategoricalFeature = 3,
        TextFeature = 4,
        Weight = 5,
        ImagePath = 6,
        SamplingKey = 7,
        UserId = 8,
        ItemId = 9,
        GroupId = 10
    }
}

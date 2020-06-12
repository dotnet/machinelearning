// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Samples.DataStructures
{
    public class Movie
    {
        [LoadColumn(0)]
        public string UserId;

        [LoadColumn(1)]
        public string MovieId;

        [LoadColumn(2)]
        public float Rating;
    }
}

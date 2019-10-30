// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Samples.DataStructures
{
    public class Movie
    {
        [ColumnName("userId"), LoadColumn(0)]
        public float UserId { get; set; }


        [ColumnName("movieId"), LoadColumn(1)]
        public float MovieId { get; set; }


        [ColumnName("rating"), LoadColumn(2)]
        public float Rating { get; set; }


        [ColumnName("timestamp"), LoadColumn(3)]
        public float Timestamp { get; set; }
    }
}

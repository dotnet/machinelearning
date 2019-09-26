using System;
using System.Collections.Generic;
using System.Text;
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

        [LoadColumn(3)]
        public float Timestamp;
    }
}

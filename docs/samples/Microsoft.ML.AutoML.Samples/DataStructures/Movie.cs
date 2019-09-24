using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Samples.DataStructures
{
    public class Movie
    {
        [LoadColumn(0)]
        public string userId;

        [LoadColumn(1)]
        public string movieId;

        [LoadColumn(2)]
        public float rating;

        [LoadColumn(3)]
        public float timestamp;
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    /// <summary>
    /// Loads training/validation/test sets from file
    /// </summary>
    public static class DatasetUtils
    {
        private const string DefaultTransformFormat = "Name={0}\nTransform=Linear\nSlope=1\nIntercept=0";

        public static string GetDefaultTransform(string featureName)
        {
            return string.Format(DefaultTransformFormat, featureName);
        }

        // Create feature from labels. This is required because freeform evaluations can use m:Rating
        // as a feature, for which appropriate transformations will be required.
        public static TsvFeature CreateFeatureFromRatings(short[] ratings)
        {
            // This function assumes that labels are only from 0 through 4
            // Label to feature map:
            // 4 -> 9
            // 3 -> 8
            // 2 -> 7
            // 1 -> 6
            // 0 -> 5
            // invalid -> 0
            short maxLab = ratings.Length > 0 ? ratings.Max() : (short) 0;
            IntArray ratingAsFeature = IntArray.New(
                ratings.Length, IntArrayType.Dense, IntArrayBits.Bits8, ratings.Select(x => (int) x));
            uint[] valueMap = Enumerable.Range(0, ((int) maxLab) + 1).Select(x => (uint) x + 5).ToArray();

            return new TsvFeature(ratingAsFeature, valueMap, "m:Rating");
        }

        /// <summary>
        /// Attempts to create a feature from a ulong array. The intent
        /// is that this will handle query ID.
        /// </summary>
        public static TsvFeature CreateFeatureFromQueryId(Dataset.DatasetSkeleton skel)
        {
            Dictionary<uint, int> uvalToOrder = new Dictionary<uint, int>();
            foreach (uint uintQid in skel.QueryIds.Select(qid => (uint) qid).Distinct().OrderBy(x => x))
            {
                uvalToOrder[uintQid] = uvalToOrder.Count;
            }
            IntArray bins = IntArray.New(
                skel.NumDocs, IntArrayType.Dense, IntArray.NumBitsNeeded(uvalToOrder.Count),
                skel.QueryIds.SelectMany((qid, i) =>
                    Enumerable.Repeat(uvalToOrder[(uint) qid], skel.Boundaries[i + 1] - skel.Boundaries[i])));
            uint[] valueMap = uvalToOrder.Keys.OrderBy(x => x).ToArray(uvalToOrder.Count);
            return new TsvFeature(bins, valueMap, "m:QueryId");
        }
    }
}
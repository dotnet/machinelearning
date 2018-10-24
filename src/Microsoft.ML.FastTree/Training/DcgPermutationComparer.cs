// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using System.Collections.Generic;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
    public abstract class DcgPermutationComparer : IComparer<int>
    {
        public abstract int Compare(int i, int j);

        public abstract double[] Scores { set; }

        public abstract short[] Labels { set; }

        public abstract int ScoresOffset { set; }

        public abstract int LabelsOffset { set; }
    }

    public static class DcgPermutationComparerFactory
    {
        public static DcgPermutationComparer GetDcgPermutationFactory(string name)
        {
            switch (name)
            {
                case "DescendingStable":
                    return new DescendingStablePermutationComparer();
                case "DescendingReverse":
                    return new DescendingReversePermutationComparer();
                case "DescendingDotNet":
                    return new DescendingDotNetPermutationComparer();
                case "DescendingStablePessimistic":
                    return new DescendingStablePessimisticPermutationComparer();
                default:
                    throw Contracts.ExceptNotSupp("DCGComparer of type '{0}' not supported", name);
            }
        }
    }

    /// <summary>
    /// Compares two integers that are indices into a vector of doubles.
    /// </summary>
    public class DescendingStablePessimisticPermutationComparer : DescendingStablePermutationComparer
    {
#pragma warning disable MSML_GeneralName // The naming is the least of this class's problems. A setter with no getter??
        protected short[] _labels;
        protected int _labelsOffset;
#pragma warning restore MSML_GeneralName

        public override short[] Labels {
            set { _labels = value; }
        }

        public override int LabelsOffset {
            set { _labelsOffset = value; }
        }

        public override int Compare(int i, int j)
        {
            if (_scores[_scoresOffset + i] > _scores[_scoresOffset + j])
                return -1;
            if (_scores[_scoresOffset + i] < _scores[_scoresOffset + j])
                return 1;
            if (_labels[_labelsOffset + i] < _labels[_labelsOffset + j])
                return -1;
            if (_labels[_labelsOffset + i] > _labels[_labelsOffset + j])
                return 1;
            return i.CompareTo(j);
        }
    }

    /// <summary>
    /// Compares two integers that are indices into a vector of doubles.
    /// </summary>
    public class DescendingStablePermutationComparer : DcgPermutationComparer
    {
#pragma warning disable MSML_GeneralName // The naming is the least of this class's problems. A setter with no getter??
        protected double[] _scores;
        protected int _scoresOffset;
#pragma warning restore MSML_GeneralName

        public override double[] Scores { set { _scores = value; } }

        public override short[] Labels { set { } }

        public override int ScoresOffset { set { _scoresOffset = value; } }

        public override int LabelsOffset { set { } }

        public override int Compare(int i, int j)
        {
            if (_scores[_scoresOffset + i] > _scores[_scoresOffset + j])
                return -1;
            if (_scores[_scoresOffset + i] < _scores[_scoresOffset + j])
                return 1;
            return i.CompareTo(j);
        }
    }

    public class DescendingReversePermutationComparer : DescendingStablePermutationComparer
    {
        public override int Compare(int i, int j)
        {
            if (_scores[_scoresOffset + i] > _scores[_scoresOffset + j])
                return -1;
            if (_scores[_scoresOffset + i] < _scores[_scoresOffset + j])
                return 1;
            return -i.CompareTo(j);
        }
    }

    public class DescendingDotNetPermutationComparer : DescendingStablePermutationComparer
    {
        public override int Compare(int i, int j)
        {
            return -_scores[_scoresOffset + i].CompareTo(_scores[_scoresOffset + j]);
        }
    }

    /// <summary>
    /// Implements an HRS based comparer to sort the ranking results for the first N results.
    /// </summary>
    public class DescendingStableIdealComparer : IComparer<int>
    {
        /// <summary>
        /// Creates an instance of the DescendingStableIdealComparer for the TOP N query/URL pairs
        /// </summary>
        /// <param name="comparefirstN">Specifies the TOP N query/URL pairs which should be used for  sorting</param>
        public DescendingStableIdealComparer(int comparefirstN)
        {
            CompareFirstN = comparefirstN;
        }

        /// <summary>
        /// Specifies the TOP N query/URL pairs which should be used for  sorting
        /// </summary>
        public int CompareFirstN { get; private set; }

        /// <summary>
        /// The HRS labels for all query/URL pairs
        /// </summary>
        public short[] Labels { get; set; }

        /// <summary>
        /// The position inside the Labels where the this query starts for which the URLs should be reshuffled.
        /// </summary>
        public int LabelsOffset { get; set; }

        /// <summary>
        /// Compare two HRS ratings for query/URL pairs
        /// </summary>
        /// <param name="x">position for query/URL pair 1</param>
        /// <param name="y">position for query/URL pair 2</param>
        /// <returns></returns>
        int IComparer<int>.Compare(int x, int y)
        {
            // sort the queries based on the ideal rating with the highest rating first
            if (Labels[LabelsOffset + x] < Labels[LabelsOffset + y])
                return 1;
            if (Labels[LabelsOffset + x] > Labels[LabelsOffset + y])
                return -1;

            // The HRS rating is the same and we do not want to change the order.
            return 0;
        }
    }
}

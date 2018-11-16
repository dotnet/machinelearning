// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Trainers.FastTree.Internal
{
#if USE_SINGLE_PRECISION
    using FloatType = System.Single;
#else
    using FloatType = System.Double;
#endif

    public enum IntArrayType { Dense, Sparse, Repeat, Segmented, Current };
    public enum IntArrayBits { Bits32 = 32, Bits16 = 16, Bits10 = 10, Bits8 = 8, Bits4 = 4, Bits1 = 1, Bits0 = 0 };

    /// <summary>
    /// An object representing an array of integers
    /// </summary>
    public abstract class IntArray : IEnumerable<int>
    {
        // The level of compression to use with features.
        // 0x1 - Use 10 bit.
        // 0x2 -
        public static int CompatibilityLevel = 0;

        /// <summary>
        /// The virtual length of the array
        /// </summary>
        public abstract int Length { get; }

        /// <summary>
        /// Returns the number of bytes written by the member ToByteArray()
        /// </summary>
        public virtual int SizeInBytes()
        {
            return 2 * sizeof(int);
        }

        /// <summary>
        /// Writes a binary representation of this class to a byte buffer, at a given position.
        /// The position is incremented to the end of the representation
        /// </summary>
        /// <param name="buffer">a byte array where the binary representation is written</param>
        /// <param name="position">the position in the byte array</param>
        public virtual void ToByteArray(byte[] buffer, ref int position)
        {
            ((int)Type).ToByteArray(buffer, ref position);
            ((int)BitsPerItem).ToByteArray(buffer, ref position);
        }

        public abstract IntArrayBits BitsPerItem { get; }

        public abstract IntArrayType Type { get; }

        public abstract MD5Hash MD5Hash { get; }

        /// <summary>
        /// Number of bytes needed to store this number of values
        /// </summary>
        public static IntArrayBits NumBitsNeeded(int numValues)
        {
            Contracts.CheckParam(numValues >= 0, nameof(numValues));
            if (numValues <= (1 << 0))
                return IntArrayBits.Bits0;
            else if (numValues <= (1 << 1))
                return IntArrayBits.Bits1;
            else if (numValues <= (1 << 4))
                return IntArrayBits.Bits4;
            else if (numValues <= (1 << 8))
                return IntArrayBits.Bits8;
            else if ((CompatibilityLevel & 1) != 0 && numValues <= (1 << 10))
                return IntArrayBits.Bits10;
            else if (numValues <= (1 << 16))
                return IntArrayBits.Bits16;
            else
                return IntArrayBits.Bits32;
        }

        public static IntArray New(int length, IntArrayType type, IntArrayBits bitsPerItem, IEnumerable<int> values)
        {
            Contracts.CheckParam(length >= 0, nameof(length));
            Contracts.CheckParam(Enum.IsDefined(typeof(IntArrayType), type) && type != IntArrayType.Current, nameof(type));
            Contracts.CheckParam(Enum.IsDefined(typeof(IntArrayBits), bitsPerItem), nameof(bitsPerItem));
            Contracts.CheckValue(values, nameof(values));

            if (type == IntArrayType.Dense || bitsPerItem == IntArrayBits.Bits0)
            {
                if (bitsPerItem == IntArrayBits.Bits0)
                {
                    Contracts.Assert(values.All(x => x == 0));
                    return new Dense0BitIntArray(length);
                }
                //else if (bitsPerItem == IntArrayBits.Bits1) return new Dense1BitIntArray(length);
                else if (bitsPerItem <= IntArrayBits.Bits4)
                    return new Dense4BitIntArray(length, values);
                else if (bitsPerItem <= IntArrayBits.Bits8)
                    return new Dense8BitIntArray(length, values);
                else if (bitsPerItem <= IntArrayBits.Bits10)
                    return new Dense10BitIntArray(length, values);
                else if (bitsPerItem <= IntArrayBits.Bits16)
                    return new Dense16BitIntArray(length, values);
                else
                    return new Dense32BitIntArray(length, values);
            }
            else if (type == IntArrayType.Sparse)
                return new DeltaSparseIntArray(length, bitsPerItem, values);
            else if (type == IntArrayType.Repeat)
                return new DeltaRepeatIntArray(length, bitsPerItem, values);
            else if (type == IntArrayType.Segmented)
                // Segmented should probably not be used in this way.
                return new SegmentIntArray(length, values);
            return null;
        }

        public static IntArray New(int length, IntArrayType type, IntArrayBits bitsPerItem)
        {
            Contracts.CheckParam(length >= 0, nameof(length));
            Contracts.CheckParam(type == IntArrayType.Current || type == IntArrayType.Repeat || type == IntArrayType.Segmented, nameof(type));

            if (type == IntArrayType.Dense || bitsPerItem == IntArrayBits.Bits0)
            {
                if (bitsPerItem == IntArrayBits.Bits0)
                    return new Dense0BitIntArray(length);
                //else if (bitsPerItem <= IntArrayBits.Bits1) return new Dense1BitIntArray(length);
                else if (bitsPerItem <= IntArrayBits.Bits4)
                    return new Dense4BitIntArray(length);
                else if (bitsPerItem <= IntArrayBits.Bits8)
                    return new Dense8BitIntArray(length);
                else if (bitsPerItem <= IntArrayBits.Bits10)
                    return new Dense10BitIntArray(length);
                else if (bitsPerItem <= IntArrayBits.Bits16)
                    return new Dense16BitIntArray(length);
                else
                    return new Dense32BitIntArray(length);
            }
            else if (type == IntArrayType.Sparse)
                return new DeltaSparseIntArray(length, bitsPerItem);
            // REVIEW: ??? What is this?
            return null;
        }

        /// <summary>
        /// Creates a new int array given a byte representation
        /// </summary>
        /// <param name="buffer">the byte array representation of the dense array. The buffer can be larger than needed since the caller might be re-using buffers from a pool</param>
        /// <param name="position">the position in the byte array</param>
        /// <returns>the int array object</returns>
        public static IntArray New(byte[] buffer, ref int position)
        {
            IntArrayType type = (IntArrayType)buffer.ToInt(ref position);
            IntArrayBits bitsPerItem = (IntArrayBits)buffer.ToInt(ref position);

            if (type == IntArrayType.Dense)
            {
                if (bitsPerItem == IntArrayBits.Bits0)
                    return new Dense0BitIntArray(buffer, ref position);
                else if (bitsPerItem == IntArrayBits.Bits4)
                    return new Dense4BitIntArray(buffer, ref position);
                else if (bitsPerItem == IntArrayBits.Bits8)
                    return new Dense8BitIntArray(buffer, ref position);
                else if (bitsPerItem == IntArrayBits.Bits10)
                    return new Dense10BitIntArray(buffer, ref position);
                else if (bitsPerItem == IntArrayBits.Bits16)
                    return new Dense16BitIntArray(buffer, ref position);
                else
                    return new Dense32BitIntArray(buffer, ref position);
            }
            else if (type == IntArrayType.Sparse)
                return new DeltaSparseIntArray(buffer, ref position);
            else if (type == IntArrayType.Repeat)
                return new DeltaRepeatIntArray(buffer, ref position);
            else if (type == IntArrayType.Segmented)
                return new SegmentIntArray(buffer, ref position);
            return null;
        }

        /// <summary>
        /// Clones the contents of this IntArray into an new IntArray
        /// </summary>
        /// <param name="bitsPerItem">The number of bits per item in the created IntArray</param>
        /// <param name="type">The type of the new IntArray</param>
        public abstract IntArray Clone(IntArrayBits bitsPerItem, IntArrayType type);

        /// <summary>
        /// Clone an IntArray containing only the items indexed by <paramref name="itemIndices"/>
        /// </summary>
        /// <param name="itemIndices"> item indices will be contained in the cloned IntArray  </param>
        /// <returns> The cloned IntArray </returns>
        public abstract IntArray Clone(int[] itemIndices);

        public abstract IntArray[] Split(int[][] assignment);

        /// <summary>
        /// Gets an indexer into the array
        /// </summary>
        /// <returns>An indexer into the array</returns>
        public abstract IIntArrayForwardIndexer GetIndexer();

        public virtual void Sumup(SumupInputData input, FeatureHistogram histogram)
        {
            Contracts.Assert((input.Weights == null) == (histogram.SumWeightsByBin == null));
            if (histogram.SumWeightsByBin != null)
            {
                SumupWeighted(input, histogram);
                return;
            }
            IIntArrayForwardIndexer indexer = GetIndexer();
            for (int i = 0; i < input.TotalCount; i++)
            {
                int featureBin = input.DocIndices == null ? indexer[i] : indexer[input.DocIndices[i]];
                if (featureBin < 0
                    || featureBin >= histogram.SumTargetsByBin.Length
                    || featureBin >= histogram.NumFeatureValues)
                {
                    throw Contracts.Except("Feature bin {0} is invalid", featureBin);
                }

                histogram.SumTargetsByBin[featureBin] += input.Outputs[i];
                ++histogram.CountByBin[featureBin];
            }
        }

        private void SumupWeighted(SumupInputData input, FeatureHistogram histogram)
        {
            Contracts.AssertValue(histogram.SumWeightsByBin);
            Contracts.AssertValue(input.Weights);
            IIntArrayForwardIndexer indexer = GetIndexer();
            for (int i = 0; i < input.TotalCount; i++)
            {
                int featureBin = input.DocIndices == null ? indexer[i] : indexer[input.DocIndices[i]];
                if (featureBin < 0
                    || featureBin >= histogram.SumTargetsByBin.Length
                    || featureBin >= histogram.NumFeatureValues)
                {
                    throw Contracts.Except("Feature bin {0} is invalid", featureBin);
                }

                histogram.SumTargetsByBin[featureBin] += input.Outputs[i];
                histogram.SumWeightsByBin[featureBin] += input.Weights[i];
                ++histogram.CountByBin[featureBin];
            }
        }

        public abstract IEnumerator<int> GetEnumerator();

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public override int GetHashCode()
        {
            int hash = 0;
            foreach (int i in this)
                hash ^= i.GetHashCode();
            return hash;
        }

        /// <summary>
        /// Finds the most space efficient representation of the feature
        /// (with slight slack cut for dense features). The behavior of
        /// this method depends upon the static value <see cref="CompatibilityLevel"/>.
        /// </summary>
        /// <param name="workarray">Should be non-null if you want it to
        /// consider segment arrays.</param>
        /// <returns>Returns a more space efficient version of the array,
        /// or the item itself if that is impossible, somehow.</returns>
        public IntArray Compress(uint[] workarray = null)
        {
            int maxval = 0;
            int zerocount = 0;
            int runs = 0;
            int last = -1;
            int overflows = 0;
            int zoverflows = 0;
            int runnow = 0; // The longest run of having the same value.
            int len = Length;
            IIntArrayForwardIndexer ind = GetIndexer();
            for (int i = 0; i < len; ++i)
            {
                int val = ind[i];
                if (workarray != null)
                    workarray[i] = (uint)val;
                if (val == 0)
                    zerocount++;
                else if (val > maxval)
                    maxval = val;
                if (last == val)
                {
                    runs++;
                    if (++runnow > byte.MaxValue)
                    {
                        // We have 256 items in a row the same.
                        overflows++;
                        if (val == 0)
                            zoverflows++;
                        runnow = 0;
                    }
                }
                last = val;
            }
            // Estimate the costs of the available options.
            IntArrayBits classicBits = IntArray.NumBitsNeeded(maxval + 1);
            long denseBits = (long)classicBits * (long)Length;
            long sparseBits = (long)(Math.Max((int)classicBits, 8) + 8) * (long)(Length - zerocount + zoverflows);
            long rleBits = (long)(classicBits + 8) * (long)(Length - runs + overflows);
            long segBits = long.MaxValue;
            int segTransitions = 0;
            if (workarray != null)
            {
                int bits = SegmentIntArray.BitsForValue((uint)maxval);
                if (bits <= 21)
                {
                    SegmentIntArray.SegmentFindOptimalPath(workarray, Length,
                        bits, out segBits, out segTransitions);
                }
            }
            if ((IntArray.CompatibilityLevel & 0x4) == 0)
            {
                rleBits = long.MaxValue;
            }
            long bestCost = Math.Min(Math.Min(Math.Min(denseBits, sparseBits), rleBits), segBits);
            IntArrayType bestType = IntArrayType.Dense;
            if (bestCost >= denseBits * 98 / 100)
            {
                // Cut the dense bits a wee bit of slack.
            }
            else if (bestCost == sparseBits)
            {
                bestType = IntArrayType.Sparse;
            }
            else if (bestCost == rleBits)
            {
                bestType = IntArrayType.Repeat;
            }
            else
            {
                bestType = IntArrayType.Segmented;
            }
            if (bestType == Type && classicBits == BitsPerItem)
            {
                return this;
            }
            IntArray bins = null;
            if (bestType != IntArrayType.Segmented)
            {
                bins = IntArray.New(Length, bestType, classicBits, this);
            }
            else
            {
                bins = SegmentIntArray.FromWorkArray(workarray, Length, segBits, segTransitions);
            }
            return bins;
        }
    }

    /// <summary>
    /// Interface for objects that can index into an <see cref="IntArray"/>, but only with a non-decreasing sequence of indices.
    /// </summary>
    public interface IIntArrayForwardIndexer
    {
        /// <summary>
        /// Gets the element at the given index.
        /// </summary>
        /// <param name="index">Index to get</param>
        /// <returns>The value at the index</returns>
        int this[int index] { get; }
    }
}

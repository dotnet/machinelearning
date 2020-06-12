// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Internal.Internallearn
{
    /// <summary>
    /// Drops slots from a fixed or variable sized column based on slot ranges.
    /// </summary>
    [BestFriend]
    internal sealed class SlotDropper
    {
        private readonly int[] _lengthReduction;
        private readonly int _srcLength;

        /// <summary>
        /// Returns -1 for non vector and unknown length vectors.
        /// </summary>
        public int DstLength { get; }

        public int[] SlotsMin { get; }

        public int[] SlotsMax { get; }

        /// <summary>
        /// Constructs slot dropper. It expects the slot ranges to be in sorted order and not overlap.
        /// </summary>
        /// <param name="srcLength">0 indicates variable sized vector.</param>
        /// <param name="slotsMin">Low limit of ranges to be dropped.</param>
        /// <param name="slotsMax">Upper limit of ranges to be dropped. </param>
        public SlotDropper(int srcLength, int[] slotsMin, int[] slotsMax)
        {
            Contracts.CheckValue(slotsMin, nameof(slotsMin));
            Contracts.CheckValue(slotsMax, nameof(slotsMax));
            Contracts.CheckNonEmpty(slotsMin, nameof(slotsMin));
            Contracts.CheckNonEmpty(slotsMax, nameof(slotsMax));
            Contracts.CheckParam(slotsMin.Length == slotsMax.Length,
                nameof(slotsMin), nameof(slotsMin) + " and " + nameof(slotsMax) + " are not equal length");

            SlotsMin = slotsMin;
            SlotsMax = slotsMax;
            _srcLength = srcLength;
            _lengthReduction = ComputeLengthReduction();

            Contracts.Check(SlotsMin.Length == _lengthReduction.Length);

            DstLength = srcLength > 1 ? ComputeLength(srcLength) : 0;
        }

        private int ComputeLength(int srcLength)
        {
            int index = SlotsMin.FindIndexSorted(srcLength);
            if (index == 0)
                return srcLength;
            index--;
            int dstLength = srcLength - _lengthReduction[index] + Math.Max(SlotsMax[index] - srcLength + 1, 0);
            Contracts.Assert(dstLength >= 0);
            return dstLength;
        }

        private int[] ComputeLengthReduction()
        {
            int[] lengthReduction = new int[SlotsMax.Length];
            int lengthRed = 0;
            int prevLim = -1;
            for (int i = 0; i < SlotsMax.Length; i++)
            {
                Contracts.Assert(SlotsMin[i] > prevLim);
                Contracts.Assert(SlotsMin[i] <= SlotsMax[i]);
                prevLim = SlotsMax[i] + 1;
                lengthRed += prevLim - SlotsMin[i];
                lengthReduction[i] = lengthRed;
            }

            return lengthReduction;
        }

        /// <summary>
        /// Returns a getter that drops slots.
        /// </summary>
        public ValueGetter<VBuffer<T>> SubsetGetter<T>(ValueGetter<VBuffer<T>> getter)
        {
            return
                (ref VBuffer<T> src) =>
                {
                    getter(ref src);
                    DropSlots(ref src, ref src);
                };
        }

        /// <summary>
        /// Drops slots from src and populates the dst with the resulting vector. Slots are
        /// dropped based on min and max slots that were passed at the constructor.
        /// </summary>
        public void DropSlots<TDst>(ref VBuffer<TDst> src, ref VBuffer<TDst> dst)
        {
            if (src.Length <= SlotsMin[0])
            {
                // There is nothing to drop, just swap buffers.
                Utils.Swap(ref src, ref dst);
                return;
            }

            int newLength = DstLength == 0 ? ComputeLength(src.Length) : DstLength;
            if (newLength == 0)
            {
                // All slots dropped.
                VBufferUtils.Resize(ref dst, 1, 0);
                return;
            }

            Contracts.Assert(newLength < src.Length);

            // End of the trivial cases
            // At this point, we need to drop some slots and keep some slots.
            VBufferEditor<TDst> editor;
            var srcValues = src.GetValues();
            if (src.IsDense)
            {
                editor = VBufferEditor.Create(ref dst, newLength);

                int iDst = 0;
                int iSrc = 0;
                for (int i = 0; i < SlotsMax.Length && iSrc < src.Length; i++)
                {
                    var lim = Math.Min(SlotsMin[i], src.Length);
                    while (iSrc < lim)
                    {
                        Contracts.Assert(iDst <= iSrc);
                        editor.Values[iDst++] = srcValues[iSrc++];
                    }
                    iSrc = SlotsMax[i] + 1;
                }
                while (iSrc < src.Length)
                {
                    Contracts.Assert(iDst <= iSrc);
                    editor.Values[iDst++] = srcValues[iSrc++];
                }
                Contracts.Assert(iDst == newLength);
                dst = editor.Commit();
                return;
            }

            // Sparse case.
            // Approximate new count is min(#indices, newLength).
            var newCount = Math.Min(srcValues.Length, newLength);
            var indices = dst.GetIndices();
            var srcIndices = src.GetIndices();

            Contracts.Assert(newCount <= src.Length);

            editor = VBufferEditor.Create(
                ref dst,
                newLength,
                newCount,
                requireIndicesOnDense: true);

            int iiDst = 0;
            int iiSrc = 0;
            int iOffset = 0;
            int iRange = 0;
            int min = SlotsMin[iRange];
            // REVIEW: Consider using a BitArray with the slots to keep instead of SlotsMax. It would
            // only make sense when the number of ranges is greater than the number of slots divided by 32.
            int max = SlotsMax[iRange];
            while (iiSrc < srcValues.Length)
            {
                // Copy (with offset) the elements before the current range.
                var index = srcIndices[iiSrc];
                if (index < min)
                {
                    Contracts.Assert(iiDst <= iiSrc);
                    editor.Indices[iiDst] = index - iOffset;
                    editor.Values[iiDst++] = srcValues[iiSrc++];
                    continue;
                }
                if (index <= max)
                {
                    // Skip elements in the current range.
                    iiSrc++;
                    continue;
                }

                // Find the next range.
                const int threshold1 = 20;
                const int threshold2 = 10;
                while (++iRange < SlotsMax.Length && SlotsMax[iRange] < index)
                {
                    if (SlotsMax.Length - iRange >= threshold1 &&
                        SlotsMax[iRange + threshold2] < index)
                    {
                        iRange = SlotsMax.FindIndexSorted(iRange + threshold2, SlotsMax.Length, index);
                        Contracts.Assert(iRange == SlotsMax.Length ||
                                         iRange > 0 && SlotsMax[iRange - 1] < index && index <= SlotsMax[iRange]);
                        break;
                    }
                }
                if (iRange < SlotsMax.Length)
                {
                    min = SlotsMin[iRange];
                    max = SlotsMax[iRange];
                }
                else
                    min = max = src.Length;
                if (iRange > 0)
                    iOffset = _lengthReduction[iRange - 1];
                Contracts.Assert(index <= max);
            }

            dst = editor.CommitTruncated(iiDst);
        }

        public IEnumerable<long> GetPreservedSlots()
        {
            var slots = Enumerable.Range(0, _srcLength);
            var droppedSlots = Enumerable.Range(SlotsMin[0], SlotsMax[0] - SlotsMin[0] + 1);
            for (int i = 1; i < SlotsMin.Length; i++)
            {
                droppedSlots = droppedSlots.Concat(Enumerable.Range(SlotsMin[i], SlotsMax[i] - SlotsMin[i] + 1));
            }
            return slots.Except(droppedSlots).Select(i=>(long)i);
        }
    }
}

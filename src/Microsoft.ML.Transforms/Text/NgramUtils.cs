// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// This delegate represents a function that gets an ngram as input, and outputs the id of
    /// the ngram and whether or not to continue processing ngrams.
    /// </summary>
    /// <param name="ngram">The array containing the ngram</param>
    /// <param name="lim">The ngram is stored in ngram[0],...ngram[lim-1].</param>
    /// <param name="icol">The index of the column the transform is applied to.</param>
    /// <param name="more">True if processing should continue, false if it should stop.
    /// It is true on input, so only needs to be set to false.</param>
    /// <returns>The ngram slot if it was found, -1 otherwise.</returns>
    internal delegate int NgramIdFinder(uint[] ngram, int lim, int icol, ref bool more);

    // A class that given a VBuffer of keys, finds all the ngrams in it, and maintains a vector of ngram-counts.
    // The id of each ngram is found by calling an NgramIdFinder delegate. This class can also be used to build
    // an ngram dictionary, by defining an NgramIdFinder that adds the ngrams to a dictionary and always return false.
    internal sealed class NgramBufferBuilder
    {
        // This buffer builder maintains the vector of ngram-counts.
        private readonly BufferBuilder<Float> _bldr;
        // A queue that holds _ngramLength+_skipLength keys, so that it contains all the ngrams starting with the
        // first key in the ngram.
        private readonly FixedSizeQueue<uint> _queue;
        // The maximum ngram length.
        private readonly int _ngramLength;
        // The maximum number of skips contained in an ngram.
        private readonly int _skipLength;
        // An array of length _ngramLength, containing the current ngram.
        private readonly uint[] _ngram;
        // The maximum ngram id.
        private readonly int _slotLim;
        private readonly NgramIdFinder _finder;

        public const int MaxSkipNgramLength = 10;

        public bool IsEmpty { get { return _slotLim == 0; } }

        public NgramBufferBuilder(int ngramLength, int skipLength, int slotLim, NgramIdFinder finder)
        {
            Contracts.Assert(ngramLength > 0);
            Contracts.Assert(skipLength >= 0);
            Contracts.Assert(ngramLength <= MaxSkipNgramLength - skipLength);
            Contracts.Assert(slotLim >= 0);

            _ngramLength = ngramLength;
            _skipLength = skipLength;
            _slotLim = slotLim;

            _ngram = new uint[_ngramLength];
            _queue = new FixedSizeQueue<uint>(_ngramLength + _skipLength);
            _bldr = BufferBuilder<Float>.CreateDefault();
            _finder = finder;
        }

        public void Reset()
        {
            _bldr.Reset(_slotLim, false);
            _queue.Clear();
        }

        public bool AddNgrams(ref VBuffer<uint> src, int icol, uint keyMax)
        {
            Contracts.Assert(icol >= 0);
            Contracts.Assert(keyMax > 0);

            uint curKey = 0;
            if (src.IsDense)
            {
                for (int i = 0; i < src.Length; i++)
                {
                    curKey = src.Values[i];
                    if (curKey > keyMax)
                        curKey = 0;

                    _queue.AddLast(curKey);

                    // Add the ngram counts
                    if (_queue.IsFull && !ProcessNgrams(icol))
                        return false;
                }
            }
            else
            {
                var queueSize = _queue.Capacity;

                int iindex = 0;
                for (int i = 0; i < src.Length; i++)
                {
                    if (iindex < src.Count && i == src.Indices[iindex])
                    {
                        curKey = src.Values[iindex];
                        if (curKey > keyMax)
                            curKey = 0;
                        iindex++;
                    }
                    else
                        curKey = 0;

                    _queue.AddLast(curKey);
                    if (!_queue.IsFull)
                        continue;

                    // Add the ngram counts
                    if (!ProcessNgrams(icol))
                        return false;
                }
            }

            if (_queue.IsFull)
                _queue.RemoveFirst();

            // Process the grams of the remaining terms
            while (_queue.Count > 0)
            {
                if (!ProcessNgrams(icol))
                    return false;
                _queue.RemoveFirst();
            }
            return true;
        }

        public void GetResult(ref VBuffer<Float> dst)
        {
            _bldr.GetResult(ref dst);
        }

        // Returns false if there is no need to process more ngrams. 
        private bool ProcessNgrams(int icol)
        {
            Contracts.Assert(_queue.Count > 0);

            _ngram[0] = _queue[0];

            int slot;
            bool more = true;
            if ((slot = _finder(_ngram, 1, icol, ref more)) >= 0)
            {
                Contracts.Assert(0 <= slot && slot < _slotLim);
                _bldr.AddFeature(slot, 1);
            }

            if (_queue.Count == 1 || !more)
                return more;

            if (_skipLength > 0)
                return ProcessSkipNgrams(icol, 1, 0);

            for (int i = 1; i < _queue.Count; i++)
            {
                _ngram[i] = _queue[i];
                Contracts.Assert(more);
                if ((slot = _finder(_ngram, i + 1, icol, ref more)) >= 0)
                {
                    Contracts.Assert(0 <= slot && slot < _slotLim);
                    _bldr.AddFeature(slot, 1);
                }
                if (!more)
                    return false;
            }

            return true;
        }

        // Uses DFS. When called with i and skips, it assumes that the 
        // first i terms in the _ngram array are already populated using "skips" skips, 
        // and it adds the (i+1)st term. It then recursively calls ProcessSkipNgrams
        // to add the next term.
        private bool ProcessSkipNgrams(int icol, int i, int skips)
        {
            Contracts.Assert(0 < i && i < _ngramLength);
            Contracts.Assert(0 <= skips && skips <= _skipLength);
            Contracts.Assert(i + skips < _queue.Count);
            Contracts.Assert(i > 1 || skips == 0);
            Contracts.Assert(_ngram.Length == _ngramLength);

            bool more = true;
            for (int k = skips; k <= _skipLength && k + i < _queue.Count; k++)
            {
                _ngram[i] = _queue[k + i];
                int slot;
                Contracts.Assert(more);
                if ((slot = _finder(_ngram, i + 1, icol, ref more)) >= 0)
                {
                    Contracts.Assert(0 <= slot && slot < _slotLim);
                    _bldr.AddFeature(slot, 1);
                }
                if (!more || (i + 1 < _ngramLength && i + k + 1 < _queue.Count && !ProcessSkipNgrams(icol, i + 1, k)))
                    return false;
            }
            return more;
        }
    }
}

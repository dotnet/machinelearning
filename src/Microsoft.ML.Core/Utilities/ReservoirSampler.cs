// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    /// <summary>
    /// This is an interface for creating samples of a requested size from a stream of data of type <typeparamref name="T"/>. 
    /// The sample is created in one pass by calling <see cref="Sample"/> for every data point in the stream. Implementations should have 
    /// a delegate for getting the next data point, which is invoked if the current data point should go into the reservoir.
    /// </summary>
    public interface IReservoirSampler<T>
    {
        /// <summary>
        /// If the number of elements sampled is less than the reservoir size, this should return the number of elements sampled.
        /// Otherwise it should return the reservoir size.
        /// </summary>
        int Size { get; }

        /// <summary>
        /// Returns the number of elements sampled so far.
        /// </summary>
        long NumSampled { get; }

        /// <summary>
        /// Sample the next data point from the stream.
        /// </summary>
        void Sample();

        /// <summary>
        /// This must be called before any calls to <see cref="GetSample"/>, and no subsequent calls to <see cref="Sample"/> can
        /// be made after that.
        /// </summary>
        void Lock();

        /// <summary>
        /// Return the elements in the sample.
        /// </summary>
        IEnumerable<T> GetSample();
    }

    /// <summary>
    /// This class produces a sample without replacement from a stream of data of type <typeparamref name="T"/>. 
    /// It is instantiated with a delegate that gets the next data point, and builds a reservoir in one pass by calling <see cref="Sample"/> 
    /// for every data point in the stream. In case the next data point does not get 'picked' into the reservoir, the delegate is not invoked.
    /// Sampling is done according to the algorithm in this paper: <see href="http://epubs.siam.org/doi/pdf/10.1137/1.9781611972740.53"/>.
    /// </summary>
    public sealed class ReservoirSamplerWithoutReplacement<T> : IReservoirSampler<T>
    {
        // This array contains a cache of the elements composing the reservoir.
        private readonly T[] _cache;

        private readonly IRandom _rnd;

        private long _numSampled;
        private readonly ValueGetter<T> _getter;

        private bool _locked;

        public int Size { get { return _numSampled <= _cache.Length ? (int)_numSampled : _cache.Length; } }

        public long NumSampled { get { return _numSampled; } }

        public ReservoirSamplerWithoutReplacement(IRandom rnd, int size, ValueGetter<T> getter)
        {
            Contracts.CheckValue(rnd, nameof(rnd));
            Contracts.CheckParam(size > 0, nameof(size), "Reservoir size must be positive");
            Contracts.CheckValue(getter, nameof(getter));

            _rnd = rnd;
            _cache = new T[size];
            _getter = getter;
        }

        public void Sample()
        {
            if (_locked)
                throw Contracts.Except("Cannot continue to sample after Lock() has been called");

            _numSampled++;

            // If the number of samples seen so far is less than the total reservoir size, cache the new sample.
            if (_numSampled <= _cache.Length)
                _getter(ref _cache[_numSampled - 1]);
            else if (_rnd.NextDouble() * _numSampled < _cache.Length)
            {
                // Replace a random existing sample with a new sample.
                int ind = _rnd.Next(_cache.Length);
                _getter(ref _cache[ind]);
            }
        }

        public void Lock()
        {
            _locked = true;
        }

        /// <summary>
        /// Gets the reservoir sample.
        /// </summary>
        public IEnumerable<T> GetSample()
        {
            if (!_locked)
                throw Contracts.Except("Call Lock() before the call to GetSample()");

            for (int i = 0; i < _numSampled; i++)
            {
                if (i == _cache.Length)
                    yield break;
                yield return _cache[i];
            }
        }
    }

    /// <summary>
    /// This class produces a sample with replacement from a stream of data of type <typeparamref name="T"/>. 
    /// It is instantiated with a delegate that gets the next data point, and builds a reservoir in one pass by calling <see cref="Sample"/> 
    /// for every data point in the stream. In case the next data point does not get 'picked' into the reservoir, the delegate is not invoked.
    /// Sampling is done according to the algorithm in this paper: <see href="http://epubs.siam.org/doi/pdf/10.1137/1.9781611972740.53"/>.
    /// </summary>
    public sealed class ReservoirSamplerWithReplacement<T> : IReservoirSampler<T>
    {
        // This array contains pointers to the elements in the _cache array that are currently in the reservoir (may contain duplicates).
        private readonly int[] _reservoir;

        // This array contains a cache of the elements composing the reservoir. The _counts array is parallel to it, and contains the
        // number of times each of these elements appears in the reservoir.
        private readonly T[] _cache;
        private readonly int[] _counts;

        private readonly IRandom _rnd;

        private long _numSampled;
        private readonly ValueGetter<T> _getter;

        private bool _locked;

        public int Size { get { return _numSampled <= _cache.Length ? (int)_numSampled : _cache.Length; } }

        public long NumSampled { get { return _numSampled; } }

        public ReservoirSamplerWithReplacement(IRandom rnd, int size, ValueGetter<T> getter)
        {
            Contracts.CheckValue(rnd, nameof(rnd));
            Contracts.CheckParam(size > 0, nameof(size), "Reservoir size must be positive");
            Contracts.CheckValue(getter, nameof(getter));

            _rnd = rnd;
            _cache = new T[size];
            _counts = new int[size];
            _reservoir = new int[size];
            _getter = getter;
        }

        public void Sample()
        {
            Contracts.Check(!_locked, "Cannot continue to sample after Lock() has been called");

            _numSampled++;

            // If the number of samples seen so far is less than the total reservoir size, cache the new sample.
            if (_numSampled <= _cache.Length)
            {
                _getter(ref _cache[_numSampled - 1]);

                // If the cache is full, sample from it with replacement into the reservoir.
                if (_numSampled == _cache.Length)
                {
                    for (int i = 0; i < _cache.Length; i++)
                    {
                        _reservoir[i] = _rnd.Next(_cache.Length);
                        _counts[_reservoir[i]]++;
                    }
                }
            }
            else
            {
                // Choose how many times to insert the current instance into the reservoir.
                int k = Stats.SampleFromBinomial(_rnd, _cache.Length, 1.0 / _numSampled);

                if (k > 0)
                {
                    int ind = _rnd.Next(_reservoir.Length);
                    // If the item referenced at this index appears more than once in the reservoir, we cannot overwrite it,
                    // we need to find an available place in the cache.
                    if (_counts[_reservoir[ind]] > 1)
                    {
                        Contracts.Assert(_counts.Contains(0));
                        var tmp = _counts.Select((count, i) => new KeyValuePair<int, int>(count, i)).First(kvp => kvp.Key == 0);
                        _counts[_reservoir[ind]]--;
                        _reservoir[ind] = tmp.Value;
                        _counts[tmp.Value] = 1;
                    }
                    else if (_counts[_reservoir[ind]] == 0)
                        _counts[_reservoir[ind]]++;
                    _getter(ref _cache[_reservoir[ind]]);

                    for (int j = 1; j < k; )
                    {
                        var next = _rnd.Next(_reservoir.Length);
                        if (_reservoir[next] == _reservoir[ind])
                            continue;
                        _counts[_reservoir[next]]--;
                        _reservoir[next] = _reservoir[ind];
                        _counts[_reservoir[next]]++;
                        j++;
                    }
                }
            }
        }

        /// <summary>
        /// Returns the cache. Users should not change the elements of the returned array.
        /// Access only elements up to the logical length of the array, which is <see cref="Size"/>.
        /// </summary>
        public T[] GetCache()
        {
            return _cache;
        }

        public void Lock()
        {
            if (!_locked && _numSampled < _reservoir.Length)
            {
                // The reservoir is still just a cache, sample from it with replacement.
                for (int i = 0; i < _cache.Length; i++)
                {
                    _reservoir[i] = _rnd.Next((int)_numSampled);
                    _counts[_reservoir[i]]++;
                }
            }
            _locked = true;
        }

        /// <summary>
        /// Gets a reservoir sample with replacement of the elements sampled so far. Users should not change the 
        /// elements returned since multiple elements in the reservoir might be pointing to the same memory.
        /// </summary>
        public IEnumerable<T> GetSample()
        {
            Contracts.Check(_locked, "Call Lock() before the call to GetSample()");

            foreach (var ind in _reservoir)
                yield return _cache[ind];
        }
    }
}

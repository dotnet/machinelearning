// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    /// <summary>
    /// Implements a heap.
    /// </summary>
    public sealed class Heap<T>
    {
        private readonly List<T> _rgv; // The heap elements. The 0th element is a dummy.
        private readonly Func<T, T, bool> _fnReverse;

        /// <summary>
        /// A Heap structure gives efficient access to the ordered next element.
        /// </summary>
        /// <param name="fnReverse">A delegate that takes two <c>T</c> objects, and if
        /// it returns true then the second object should be popped before the first</param>
        public Heap(Func<T, T, bool> fnReverse)
        {
            Contracts.AssertValue(fnReverse);

            _rgv = new List<T>();
            _rgv.Add(default(T));
            _fnReverse = fnReverse;

            AssertValid();
        }

        /// <summary>
        /// A Heap structure gives efficient access to the ordered next element.
        /// </summary>
        /// <param name="fnReverse">A delegate that takes two <c>T</c> objects, and if
        /// it returns true then the second object should be popped before the first</param>
        /// <param name="capacity">The initial capacity of the heap</param>
        public Heap(Func<T, T, bool> fnReverse, int capacity)
        {
            Contracts.AssertValue(fnReverse);
            Contracts.Assert(capacity >= 0);

            _rgv = new List<T>(capacity);
            _rgv.Add(default(T));
            _fnReverse = fnReverse;

            AssertValid();
        }

        [Conditional("DEBUG")]
        private void AssertValid()
        {
            Contracts.AssertValue(_fnReverse);
            Contracts.AssertValue(_rgv);
            Contracts.Assert(_rgv.Count > 0);
        }

        /// <summary> Func tests true if first element should be after the second
        /// </summary>
        public Func<T, T, bool> FnReverse
        {
            get
            {
                AssertValid();
                return _fnReverse;
            }
        }

        /// <summary>
        /// Current count of elements remaining in the heap
        /// </summary>
        public int Count
        {
            get
            {
                AssertValid();
                return _rgv.Count - 1;
            }
        }

        private static int Parent(int iv)
        {
            return iv >> 1;
        }
        private static int Left(int iv)
        {
            return iv + iv;
        }
        private static int Right(int iv)
        {
            return iv + iv + 1;
        }

        // For tracking indices for items.
        private void MoveTo(T v, int iv)
        {
            Contracts.Assert(iv > 0);

            _rgv[iv] = v;
        }

        /// <summary>
        /// Discard all elements currently in the heap
        /// </summary>
        public void Clear()
        {
            AssertValid();
            _rgv.RemoveRange(1, _rgv.Count - 1);
            AssertValid();
        }

        /// <summary>
        /// Peek at the first element in the heap
        /// </summary>
        public T Top
        {
            get
            {
                AssertValid();
                if (_rgv.Count <= 1)
                    return default(T);
                return _rgv[1];
            }
        }

        /// <summary>
        /// Remove and return the first element in the heap
        /// </summary>
        /// <returns>The first element in the heap</returns>
        public T Pop()
        {
            AssertValid();

            int cv = _rgv.Count;
            Contracts.Check(cv > 1);

            T vRes = _rgv[1];
            _rgv[1] = _rgv[--cv];
            _rgv.RemoveAt(cv);
            if (cv > 1)
                BubbleDown(1);

            AssertValid();
            return vRes;
        }

        /// <summary>
        /// Add a new element to the heap
        /// </summary>
        /// <param name="item">The item to add</param>
        public void Add(T item)
        {
            AssertValid();

            int iv = _rgv.Count;
            _rgv.Add(item);
            BubbleUp(iv);

            AssertValid();
        }

        private void BubbleUp(int iv)
        {
            Contracts.Assert(0 < iv && iv < _rgv.Count);

            T v = _rgv[iv];
            int ivPar;
            for (; (ivPar = Parent(iv)) > 0 && _fnReverse(_rgv[ivPar], v); iv = ivPar)
                MoveTo(_rgv[ivPar], iv);
            MoveTo(v, iv);
        }

        private void BubbleDown(int iv)
        {
            Contracts.Assert(0 < iv && iv < _rgv.Count);

            int cv = _rgv.Count;
            T v = _rgv[iv];
            int ivChild;
            for (; (ivChild = Left(iv)) < cv; iv = ivChild)
            {
                if (ivChild + 1 < cv && _fnReverse(_rgv[ivChild], _rgv[ivChild + 1]))
                    ivChild++;
                if (!_fnReverse(v, _rgv[ivChild]))
                    break;
                MoveTo(_rgv[ivChild], iv);
            }
            MoveTo(v, iv);
        }
    }

    /// <summary>
    /// For the heap to allow deletion, the heap node has to derive from this class.
    /// </summary>
    public abstract partial class HeapNode
    {
        // Where this node lives in the heap. Zero means it isn't in the heap.
        private int _index;

        protected HeapNode()
        {
            Contracts.Assert(!InHeap);
        }

        public bool InHeap { get { return _index > 0; } }
    }

    public abstract partial class HeapNode
    {
        /// <summary>
        /// Implements a heap.
        /// </summary>
        public sealed class Heap<T>
            where T : HeapNode
        {
            private readonly List<T> _rgv; // The heap elements. The 0th element is a dummy.
            private readonly Func<T, T, bool> _fnReverse;

            /// <summary>
            /// A Heap structure gives efficient access to the ordered next element.
            /// </summary>
            /// <param name="fnReverse">A delegate that takes two <c>T</c> objects, and if
            /// it returns true then the second object should be popped before the first</param>
            public Heap(Func<T, T, bool> fnReverse)
            {
                Contracts.AssertValue(fnReverse);

                _rgv = new List<T>();
                _rgv.Add(default(T));
                _fnReverse = fnReverse;

                AssertValid();
            }

            /// <summary>
            /// A Heap structure gives efficient access to the ordered next element.
            /// </summary>
            /// <param name="fnReverse">A delegate that takes two <c>T</c> objects, and if
            /// it returns true then the second object should be popped before the first</param>
            /// <param name="capacity">The initial capacity of the heap</param>
            public Heap(Func<T, T, bool> fnReverse, int capacity)
            {
                Contracts.AssertValue(fnReverse);
                Contracts.Assert(capacity >= 0);

                _rgv = new List<T>(capacity);
                _rgv.Add(default(T));
                _fnReverse = fnReverse;

                AssertValid();
            }

            [Conditional("DEBUG")]
            private void AssertValid()
            {
                Contracts.AssertValue(_fnReverse);
                Contracts.AssertValue(_rgv);
                Contracts.Assert(_rgv.Count > 0);
            }

            /// <summary> Func tests true if first element should be after the second
            /// </summary>
            public Func<T, T, bool> FnReverse
            {
                get
                {
                    AssertValid();
                    return _fnReverse;
                }
            }

            /// <summary>
            /// Current count of elements remaining in the heap
            /// </summary>
            public int Count
            {
                get
                {
                    AssertValid();
                    return _rgv.Count - 1;
                }
            }

            private static int Parent(int iv)
            {
                return iv >> 1;
            }
            private static int Left(int iv)
            {
                return iv + iv;
            }
            private static int Right(int iv)
            {
                return iv + iv + 1;
            }

            // For tracking indices for items.
            private void MoveTo(T v, int iv)
            {
                Contracts.Assert(iv > 0);

                _rgv[iv] = v;
                v._index = iv;
            }

            private T Get(int index)
            {
                Contracts.Assert(0 < index && index < _rgv.Count);
                var result = _rgv[index];
                Contracts.Assert(result._index == index);
                return result;
            }

            /// <summary>
            /// Remove all elements currently in the heap
            /// </summary>
            public void Clear()
            {
                AssertValid();
                int cv = _rgv.Count;
                for (int i = 1; i < cv; i++)
                {
                    var v = Get(i);
                    v._index = 0;
                }
                _rgv.RemoveRange(1, cv - 1);
                AssertValid();
            }

            /// <summary>
            /// Peek at the first element in the heap
            /// </summary>
            public T Top
            {
                get
                {
                    AssertValid();
                    if (_rgv.Count <= 1)
                        return default(T);
                    return Get(1);
                }
            }

            /// <summary>
            /// Remove and return the first element in the heap
            /// </summary>
            public T Pop()
            {
                AssertValid();

                int iv = _rgv.Count - 1;
                Contracts.Check(iv > 0);

                T vRes = Get(1);
                if (iv > 1)
                {
                    MoveTo(Get(iv), 1);
                    _rgv.RemoveAt(iv);
                    BubbleDown(1);
                }
                else
                    _rgv.RemoveAt(1);

                vRes._index = 0;
                Contracts.Assert(!vRes.InHeap);

                AssertValid();
                return vRes;
            }

            /// <summary>
            /// Add a new element to the heap
            /// </summary>
            public void Add(T item)
            {
                AssertValid();
                Contracts.AssertValue(item);
                Contracts.Check(!item.InHeap);

                int iv = _rgv.Count;
                _rgv.Add(item);
                item._index = iv;
                BubbleUp(iv);

                AssertValid();
            }

            /// <summary>
            /// Remove an element from the heap
            /// </summary>
            public void Delete(T item)
            {
                AssertValid();
                Contracts.AssertValue(item);
                Contracts.Check(item.InHeap);

                int ivDst = item._index;
                Contracts.Check(Get(ivDst) == item);

                int ivSrc = _rgv.Count - 1;
                Contracts.Assert(ivSrc >= ivDst);

                if (ivSrc > ivDst)
                {
                    MoveTo(Get(ivSrc), ivDst);
                    _rgv.RemoveAt(ivSrc);
                    BubbleDown(ivDst);
                }
                else
                    _rgv.RemoveAt(ivDst);

                item._index = 0;
                Contracts.Assert(!item.InHeap);
            }

            private void BubbleUp(int iv)
            {
                Contracts.Assert(0 < iv && iv < _rgv.Count);

                T v = Get(iv);
                int ivPar;
                for (; (ivPar = Parent(iv)) > 0 && _fnReverse(Get(ivPar), v); iv = ivPar)
                    MoveTo(Get(ivPar), iv);
                MoveTo(v, iv);
            }

            private void BubbleDown(int iv)
            {
                Contracts.Assert(0 < iv && iv < _rgv.Count);

                int cv = _rgv.Count;
                T v = Get(iv);
                int ivChild;
                for (; (ivChild = Left(iv)) < cv; iv = ivChild)
                {
                    if (ivChild + 1 < cv && _fnReverse(Get(ivChild), Get(ivChild + 1)))
                        ivChild++;
                    if (!_fnReverse(v, Get(ivChild)))
                        break;
                    MoveTo(Get(ivChild), iv);
                }
                MoveTo(v, iv);
            }
        }
    }
}

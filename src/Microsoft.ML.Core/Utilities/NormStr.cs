// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Collections;
using System.Linq;
using System.Threading;
using System.Text;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    /// <summary>
    /// Normalized string type. For string pooling.
    /// </summary>
    [BestFriend]
    internal sealed class NormStr
    {
        public readonly ReadOnlyMemory<char> Value;
        public readonly int Id;
        private readonly uint _hash;

        /// <summary>
        /// NormStr's can only be created by the Pool.
        /// </summary>
        private NormStr(ReadOnlyMemory<char> str, int id, uint hash)
        {
            Contracts.Assert(id >= 0 || id == -1 && str.IsEmpty);
            Value = str;
            Id = id;
            _hash = hash;
        }

        public override int GetHashCode()
        {
            return (int)_hash;
        }

        public sealed class Pool : IEnumerable<NormStr>
        {
            private int _mask; // Number of buckets minus 1. The number of buckets must be a power of two.
            private int[] _rgins;  // Buckets of size _mask + 1.

            // The number of strings.
            private int _cns;
            // The strings.
            private NormStr[] _rgns;
            // Parallel to _rgns. Each ulong contains the length of the string (for speedy checks) and the
            // index of the next string in the same bucket. The length is the low int and the next index
            // is the high int. Doing this gives better perf than more structured alternatives.
            private ulong[] _rgmeta;

            public int Count { get { return _cns; } }

            public Pool()
            {
                _mask = 31;
                _rgins = new int[_mask + 1];
                for (int i = 0; i < _rgins.Length; i++)
                    _rgins[i] = -1;

                AssertValid();
            }

            [Conditional("DEBUG")]
            private void AssertValid()
            {
                // Number of buckets must be a power of two.
                Contracts.AssertValue(_rgins);
                Contracts.Assert(_rgins.Length == _mask + 1);
                Contracts.Assert(Utils.IsPowerOfTwo(_mask + 1));

                Contracts.Assert(0 <= _cns & _cns <= Utils.Size(_rgns));
                Contracts.Assert(Utils.Size(_rgns) == Utils.Size(_rgmeta));
            }

            private int GetIns(uint hash)
            {
                return _rgins[(int)hash & _mask];
            }

            private int GetIins(uint hash)
            {
                return (int)hash & _mask;
            }

            /// <summary>
            /// Find the given string in the pool. If not found, returns null.
            /// </summary>
            public NormStr Get(string str, bool add = false)
            {
                AssertValid();

                if (str == null)
                    str = "";

                var strSpan = str.AsSpan();
                uint hash = Hashing.HashString(strSpan);
                int ins = GetIns(hash);
                while (ins >= 0)
                {
                    ulong meta = _rgmeta[ins];
                    if ((int)Utils.GetLo(meta) == str.Length)
                    {
                        var ns = GetNs(ins);
                        if (strSpan.SequenceEqual(ns.Value.Span))
                            return ns;
                    }
                    ins = (int)Utils.GetHi(meta);
                }
                Contracts.Assert(ins == -1);

                return add ? AddCore(str.AsMemory(), hash) : null;
            }

            public NormStr Get(ReadOnlyMemory<char> str, bool add = false)
            {
                AssertValid();

                var span = str.Span;
                uint hash = Hashing.HashString(span);
                int ins = GetIns(hash);
                while (ins >= 0)
                {
                    ulong meta = _rgmeta[ins];
                    if ((int)Utils.GetLo(meta) == str.Length)
                    {
                        var ns = GetNs(ins);
                        if (ns.Value.Span.SequenceEqual(span))
                            return ns;
                    }
                    ins = (int)Utils.GetHi(meta);
                }
                Contracts.Assert(ins == -1);

                return add ? AddCore(str, hash) : null;
            }

            /// <summary>
            /// Make sure the given string has an equivalent NormStr in the pool and return it.
            /// </summary>
            public NormStr Add(string str)
            {
                return Get(str, true);
            }

            public NormStr Add(ReadOnlyMemory<char> str)
            {
                return Get(str, true);
            }

            /// <summary>
            /// Make sure the given string has an equivalent NormStr in the pool and return it.
            /// </summary>
            public NormStr Get(StringBuilder sb, bool add = false)
            {
                AssertValid();

                if (sb == null)
                    return Get("", add);

                int cch = sb.Length;

                NormStr ns;
                uint hash = Hashing.HashString(sb);
                int ins = GetIns(hash);
                while (ins >= 0)
                {
                    ulong meta = _rgmeta[ins];
                    if ((int)Utils.GetLo(meta) == cch)
                    {
                        ns = GetNs(ins);
                        var value = ns.Value;
                        for (int ich = 0; ; ich++)
                        {
                            if (ich == cch)
                                return ns;
                            if (value.Span[ich] != sb[ich])
                                break;
                        }
                    }
                    ins = (int)Utils.GetHi(meta);
                }
                Contracts.Assert(ins == -1);

                return add ? AddCore(sb.ToString().AsMemory(), hash) : null;
            }

            /// <summary>
            /// Make sure the given string builder has an equivalent NormStr in the pool and return it.
            /// </summary>
            public NormStr Add(StringBuilder sb)
            {
                return Get(sb, true);
            }

            /// <summary>
            /// Adds the item. Does NOT check for whether the item is already present.
            /// </summary>
            private NormStr AddCore(ReadOnlyMemory<char> str, uint hash)
            {
                Contracts.Assert(str.Length >= 0);
                Contracts.Assert(Hashing.HashString(str.Span) == hash);

                if (_rgns == null)
                {
                    Contracts.Assert(_cns == 0);
                    _rgmeta = new ulong[10];
                    _rgns = new NormStr[10];
                }
                else if (_cns >= _rgns.Length)
                {
                    Contracts.Assert(_cns == _rgns.Length);
                    int size = checked(_rgns.Length / 2 + _rgns.Length);
                    Array.Resize(ref _rgmeta, size);
                    Array.Resize(ref _rgns, size);
                }
                Contracts.Assert(_cns < _rgns.Length);

                NormStr ns = new NormStr(str, _cns, hash);
                int iins = GetIins(hash);
                _rgns[_cns] = ns;
                _rgmeta[_cns] = Utils.MakeUlong((uint)_rgins[iins], (uint)ns.Value.Length);
                _rgins[iins] = _cns;

                if (++_cns >= _rgins.Length)
                    GrowTable();

                AssertValid();
                return ns;
            }

            public NormStr GetNormStrById(int id)
            {
                Contracts.CheckParam(0 <= id && id < _cns, nameof(id));
                return GetNs(id);
            }

            private NormStr GetNs(int ins)
            {
                Contracts.Assert(0 <= ins && ins < _cns);
                Contracts.Assert(_rgns[ins].Id == ins);
                return _rgns[ins];
            }

            private void GrowTable()
            {
                AssertValid();

                int size = checked(2 * _rgins.Length);
                _rgins = new int[size];
                _mask = size - 1;
                for (int i = 0; i < _rgins.Length; i++)
                    _rgins[i] = -1;

                for (int ins = 0; ins < _cns; ins++)
                {
                    var ns = GetNs(ins);
                    int iins = GetIins(ns._hash);
                    _rgmeta[ins] = Utils.MakeUlong((uint)_rgins[iins], (uint)ns.Value.Length);
                    _rgins[iins] = ins;
                }

                AssertValid();
            }

            public IEnumerator<NormStr> GetEnumerator()
            {
                for (int ins = 0; ins < _cns; ins++)
                    yield return GetNs(ins);
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }
        }
    }
}

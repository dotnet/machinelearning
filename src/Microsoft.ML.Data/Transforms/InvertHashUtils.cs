// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Data
{
    public static class InvertHashUtils
    {
        /// <summary>
        /// Clears a destination StringBuilder. If it is currently null, allocates it.
        /// </summary>
        private static void ClearDst(ref StringBuilder dst)
        {
            Contracts.AssertValueOrNull(dst);
            if (dst == null)
                dst = new StringBuilder();
            else
                dst.Clear();
        }

        /// <summary>
        /// Gets the mapping from T into a StringBuilder representation, using various heuristics.
        /// This StringBuilder representation will be a component of the composed KeyValues for the
        /// hash outputs.
        /// </summary>
        public static ValueMapper<T, StringBuilder> GetSimpleMapper<T>(ISchema schema, int col)
        {
            Contracts.AssertValue(schema);
            Contracts.Assert(0 <= col && col < schema.ColumnCount);
            var type = schema.GetColumnType(col).ItemType;
            Contracts.Assert(type.RawType == typeof(T));
            var conv = Conversion.Conversions.Instance;

            // First: if not key, then get the standard string converison.
            if (!type.IsKey)
                return conv.GetStringConversion<T>(type);

            bool identity;
            // Second choice: if key, utilize the KeyValues metadata for that key, if it has one and is text.
            if (schema.HasKeyNames(col, type.KeyCount))
            {
                // REVIEW: Non-textual KeyValues are certainly possible. Should we handle them?
                // Get the key names.
                VBuffer<ReadOnlyMemory<char>> keyValues = default;
                schema.GetMetadata(MetadataUtils.Kinds.KeyValues, col, ref keyValues);
                ReadOnlyMemory<char> value = default;

                // REVIEW: We could optimize for identity, but it's probably not worthwhile.
                var keyMapper = conv.GetStandardConversion<T, uint>(type, NumberType.U4, out identity);
                return
                    (ref T src, ref StringBuilder dst) =>
                    {
                        ClearDst(ref dst);
                        uint intermediate = 0;
                        keyMapper(ref src, ref intermediate);
                        if (intermediate == 0)
                            return;
                        keyValues.GetItemOrDefault((int)(intermediate - 1), ref value);
                        dst.AppendMemory(value);
                    };
            }

            // Third choice: just use the key value itself, subject to offsetting by the min.
            return conv.GetKeyStringConversion<T>(type.AsKey);
        }

        public static ValueMapper<KeyValuePair<int, T>, StringBuilder> GetPairMapper<T>(ValueMapper<T, StringBuilder> submap)
        {
            StringBuilder sb = null;
            char[] buffer = null;
            return
                (ref KeyValuePair<int, T> pair, ref StringBuilder dst) =>
                {
                    ClearDst(ref dst);
                    dst.Append(pair.Key);
                    dst.Append(':');
                    var subval = pair.Value;
                    submap(ref subval, ref sb);
                    AppendToEnd(sb, dst, ref buffer);
                };
        }

        public static void AppendToEnd(StringBuilder src, StringBuilder dst, ref char[] buffer)
        {
            // A direct sb -> sb copy sure would be nice...
            if (Utils.Size(src) > 0)
            {
                Utils.EnsureSize(ref buffer, src.Length);
                src.CopyTo(0, buffer, 0, src.Length);
                dst.Append(buffer, 0, src.Length);
            }
        }
    }

    public sealed class InvertHashCollector<T>
    {
        /// <summary>
        /// This is a small struct that is meant to compare akin to the value,
        /// but also maintain the order in which it was inserted, assuming that
        /// we're using something like a hashset where order is not preserved.
        /// </summary>
        private struct Pair
        {
            public readonly T Value;
            public readonly int Order;

            public Pair(T value, int order)
            {
                Contracts.Assert(order >= 0);
                Value = value;
                Order = order;
            }
        }

        private sealed class PairEqualityComparer : IEqualityComparer<Pair>
        {
            private readonly IEqualityComparer<T> _tComparer;

            public PairEqualityComparer(IEqualityComparer<T> tComparer)
            {
                _tComparer = tComparer;
            }

            public bool Equals(Pair x, Pair y)
            {
                return _tComparer.Equals(x.Value, y.Value);
            }

            public int GetHashCode(Pair obj)
            {
                return _tComparer.GetHashCode(obj.Value);
            }
        }

        // The maximum number of distinct keys to accumulate per slot.
        private readonly int _maxCount;
        // The maximum number of slots.
        private readonly int _slots;

        private readonly ValueMapper<T, StringBuilder> _stringifyMapper;
        // REVIEW: The following is very general but inefficient. If perf is a problem, then this
        // is one clear place where it should be helped.
        private readonly Dictionary<int, HashSet<Pair>> _slotToValueSet;
        private readonly IEqualityComparer<Pair> _comparer;
        private readonly ValueMapper<T, T> _copier;

        /// <summary>
        /// Constructs an invert hash collector that collects unique keys per slot, then is able
        /// to build a textual description out of that.
        /// </summary>
        /// <param name="slots">The maximum number of slots</param>
        /// <param name="maxCount">The number of distinct keys we can accumulate per slot</param>
        /// <param name="mapper">Utilized in composing the final description, once we have done
        /// collecting the distinct keys.</param>
        /// <param name="comparer">For detecting uniqueness of the keys we're collecting per slot.</param>
        /// <param name="copier">For copying input values into a value to actually store. Useful for
        /// types of objects where it is possible to do a comparison relatively quickly on some sort
        /// of "unsafe" object, but for which when we decide to actually store it we need to provide
        /// a "safe" version of the object. Utilized in the ngram hash transform, for example.</param>
        public InvertHashCollector(int slots, int maxCount, ValueMapper<T, StringBuilder> mapper,
            IEqualityComparer<T> comparer, ValueMapper<T, T> copier = null)
        {
            Contracts.Assert(slots > 0);
            Contracts.Assert(maxCount > 0);
            Contracts.AssertValue(mapper);
            Contracts.AssertValue(comparer);

            _slots = slots;
            _maxCount = maxCount;
            _stringifyMapper = mapper;
            _comparer = new PairEqualityComparer(comparer);
            _slotToValueSet = new Dictionary<int, HashSet<Pair>>();
            _copier = copier ?? ((ref T src, ref T dst) => dst = src);
        }

        private ReadOnlyMemory<char> Textify(ref StringBuilder sb, ref StringBuilder temp, ref char[] cbuffer, ref Pair[] buffer, HashSet<Pair> pairs)
        {
            Contracts.AssertValueOrNull(sb);
            Contracts.AssertValueOrNull(temp);
            Contracts.AssertValueOrNull(cbuffer);
            Contracts.AssertValueOrNull(buffer);
            Contracts.Assert(Utils.Size(pairs) > 0);
            int count = pairs.Count;

            // Keep things in the same order they were inserted, by sorting on order.
            Utils.EnsureSize(ref buffer, count);
            pairs.CopyTo(buffer);
            pairs.Clear();

            // Optimize the one value case, where we don't have to use the string builder.
            if (count == 1)
            {
                var value = buffer[0].Value;
                _stringifyMapper(ref value, ref temp);
                return Utils.Size(temp) > 0 ? temp.ToString().AsMemory() : String.Empty.AsMemory();
            }

            Array.Sort(buffer, 0, count, Comparer<Pair>.Create((x, y) => x.Order - y.Order));
            if (sb == null)
                sb = new StringBuilder();
            Contracts.Assert(sb.Length == 0);
            // The more general collision case.
            sb.Append('{');
            for (int i = 0; i < count; ++i)
            {
                var pair = buffer[i];
                if (i > 0)
                    sb.Append(',');
                var value = pair.Value;
                _stringifyMapper(ref value, ref temp);
                InvertHashUtils.AppendToEnd(temp, sb, ref cbuffer);
            }
            sb.Append('}');
            var retval = sb.ToString().AsMemory();
            sb.Clear();
            return retval;
        }

        public VBuffer<ReadOnlyMemory<char>> GetMetadata()
        {
            int count = _slotToValueSet.Count;
            Contracts.Assert(count <= _slots);
            StringBuilder sb = null;
            StringBuilder temp = null;
            Pair[] pairs = null;
            char[] cbuffer = null;

            bool sparse = count <= _slots / 2;
            if (sparse)
            {
                // Sparse
                var indices = new int[count];
                var values = new ReadOnlyMemory<char>[count];
                int i = 0;
                foreach (var p in _slotToValueSet)
                {
                    Contracts.Assert(0 <= p.Key && p.Key < _slots);
                    indices[i] = p.Key;
                    values[i++] = Textify(ref sb, ref temp, ref cbuffer, ref pairs, p.Value);
                }
                Contracts.Assert(i == count);
                Array.Sort(indices, values);
                return new VBuffer<ReadOnlyMemory<char>>((int)_slots, count, values, indices);
            }
            else
            {
                // Dense
                var values = new ReadOnlyMemory<char>[_slots];
                foreach (var p in _slotToValueSet)
                {
                    Contracts.Assert(0 <= p.Key && p.Key < _slots);
                    values[p.Key] = Textify(ref sb, ref temp, ref cbuffer, ref pairs, p.Value);
                }
                return new VBuffer<ReadOnlyMemory<char>>(values.Length, values);
            }
        }

        public void Add(int dstSlot, ValueGetter<T> getter, ref T key)
        {
            // REVIEW: I only call the getter if I determine I have to, but
            // at the cost of passing along this getter and ref argument (as opposed
            // to just the argument). Is this really appropriate or helpful?
            Contracts.Assert(0 <= dstSlot && dstSlot < _slots);
            HashSet<Pair> pairSet;
            if (_slotToValueSet.TryGetValue(dstSlot, out pairSet))
            {
                if (pairSet.Count >= _maxCount)
                    return;
            }
            else
                pairSet = _slotToValueSet[dstSlot] = new HashSet<Pair>(_comparer);
            getter(ref key);
            pairSet.Add(new Pair(key, pairSet.Count));
        }

        public void Add(int dstSlot, T key)
        {
            Contracts.Assert(0 <= dstSlot && dstSlot < _slots);
            HashSet<Pair> pairSet;
            if (_slotToValueSet.TryGetValue(dstSlot, out pairSet))
            {
                if (pairSet.Count >= _maxCount)
                    return;
            }
            else
                pairSet = _slotToValueSet[dstSlot] = new HashSet<Pair>(_comparer);
            T dst = default(T);
            _copier(ref key, ref dst);
            pairSet.Add(new Pair(dst, pairSet.Count));
        }

        public void Add(uint hash, ValueGetter<T> getter, ref T key)
        {
            // Convenience method for those where the inserters work in the hash space, not the
            // slot space, assuming that hash value of 0 gets no key.
            if (hash == 0)
                return;
            Add((int)hash - 1, getter, ref key);
        }

        public void Add(uint hash, T key)
        {
            if (hash == 0)
                return;
            Add((int)hash - 1, key);
        }
    }

    /// <summary>
    /// Simple utility class for saving a <see cref="VBuffer{T}"/> of ReadOnlyMemory
    /// as a model, both in a binary and more easily human readable form.
    /// </summary>
    public static class TextModelHelper
    {
        private const string LoaderSignature = "TextSpanBuffer";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TEXTSPBF",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        private static void Load(IChannel ch, ModelLoadContext ctx, CodecFactory factory, ref VBuffer<ReadOnlyMemory<char>> values)
        {
            Contracts.AssertValue(ch);
            ch.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // Codec parameterization: A codec parameterization that should be a ReadOnlyMemory codec
            // int: n, the number of bytes used to write the values
            // byte[n]: As encoded using the codec

            // Get the codec from the factory, and from the stream. We have to
            // attempt to read the codec from the stream, since codecs can potentially
            // be versioned based on their parameterization.
            IValueCodec codec;
            // This *could* happen if we have an old version attempt to read a new version.
            // Enabling this sort of binary classification is why we also need to write the
            // codec specification.
            if (!factory.TryReadCodec(ctx.Reader.BaseStream, out codec))
                throw ch.ExceptDecode();
            ch.AssertValue(codec);
            ch.CheckDecode(codec.Type.IsVector);
            ch.CheckDecode(codec.Type.ItemType.IsText);
            var textCodec = (IValueCodec<VBuffer<ReadOnlyMemory<char>>>)codec;

            var bufferLen = ctx.Reader.ReadInt32();
            ch.CheckDecode(bufferLen >= 0);
            using (var stream = new SubsetStream(ctx.Reader.BaseStream, bufferLen))
            {
                using (var reader = textCodec.OpenReader(stream, 1))
                {
                    reader.MoveNext();
                    values = default(VBuffer<ReadOnlyMemory<char>>);
                    reader.Get(ref values);
                }
                ch.CheckDecode(stream.ReadByte() == -1);
            }
        }

        private static void Save(IChannel ch, ModelSaveContext ctx, CodecFactory factory, ref VBuffer<ReadOnlyMemory<char>> values)
        {
            Contracts.AssertValue(ch);
            ch.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // Codec parameterization: A codec parameterization that should be a ReadOnlyMemory codec
            // int: n, the number of bytes used to write the values
            // byte[n]: As encoded using the codec

            // Get the codec from the factory
            IValueCodec codec;
            var result = factory.TryGetCodec(new VectorType(TextType.Instance), out codec);
            ch.Assert(result);
            ch.Assert(codec.Type.IsVector);
            ch.Assert(codec.Type.VectorSize == 0);
            ch.Assert(codec.Type.ItemType.RawType == typeof(ReadOnlyMemory<char>));
            IValueCodec<VBuffer<ReadOnlyMemory<char>>> textCodec = (IValueCodec<VBuffer<ReadOnlyMemory<char>>>)codec;

            factory.WriteCodec(ctx.Writer.BaseStream, codec);
            using (var mem = new MemoryStream())
            {
                using (var writer = textCodec.OpenWriter(mem))
                {
                    writer.Write(ref values);
                    writer.Commit();
                }
                ctx.Writer.WriteByteArray(mem.ToArray());
            }

            // Make this resemble, more or less, the auxiliary output from the TermTransform.
            // It will differ somewhat due to the vector being possibly sparse. To distinguish
            // between missing and empty, empties are not written at all, while missings are.
            var v = values;
            char[] buffer = null;
            ctx.SaveTextStream("Terms.txt",
                writer =>
                {
                    writer.WriteLine("# Number of terms = {0} of length {1}", v.Count, v.Length);
                    foreach (var pair in v.Items())
                    {
                        var text = pair.Value;
                        if (text.IsEmpty)
                            continue;
                        writer.Write("{0}\t", pair.Key);
                        // REVIEW: What about escaping this, *especially* for linebreaks?
                        // Do C# and .NET really have no equivalent to Python's "repr"? :(
                        if (text.IsEmpty)
                        {
                            writer.WriteLine();
                            continue;
                        }
                        Utils.EnsureSize(ref buffer, text.Length);

                        var span = text.Span;
                        for (int i = 0; i < text.Length; i++)
                            buffer[i] = span[i];

                        writer.WriteLine(buffer, 0, text.Length);
                    }
                });
        }

        public static void LoadAll(IHost host, ModelLoadContext ctx, int infoLim, out VBuffer<ReadOnlyMemory<char>>[] keyValues, out ColumnType[] kvTypes)
        {
            Contracts.AssertValue(host);
            host.AssertValue(ctx);

            using (var ch = host.Start("LoadTextValues"))
            {
                // Try to find the key names.
                VBuffer<ReadOnlyMemory<char>>[] keyValuesLocal = null;
                ColumnType[] kvTypesLocal = null;
                CodecFactory factory = null;
                const string dirFormat = "Vocabulary_{0:000}";
                for (int iinfo = 0; iinfo < infoLim; iinfo++)
                {
                    ctx.TryProcessSubModel(string.Format(dirFormat, iinfo),
                        c =>
                        {
                            // Load the lazily initialized structures, if needed.
                            if (keyValuesLocal == null)
                            {
                                keyValuesLocal = new VBuffer<ReadOnlyMemory<char>>[infoLim];
                                kvTypesLocal = new ColumnType[infoLim];
                                factory = new CodecFactory(host);
                            }
                            Load(ch, c, factory, ref keyValuesLocal[iinfo]);
                            kvTypesLocal[iinfo] = new VectorType(TextType.Instance, keyValuesLocal[iinfo].Length);
                        });
                }

                keyValues = keyValuesLocal;
                kvTypes = kvTypesLocal;
                ch.Done();
            }
        }

        public static void SaveAll(IHost host, ModelSaveContext ctx, int infoLim, VBuffer<ReadOnlyMemory<char>>[] keyValues)
        {
            Contracts.AssertValue(host);
            host.AssertValue(ctx);
            host.AssertValueOrNull(keyValues);

            if (keyValues == null)
                return;

            using (var ch = host.Start("SaveTextValues"))
            {
                // Save the key names as separate submodels.
                const string dirFormat = "Vocabulary_{0:000}";
                CodecFactory factory = new CodecFactory(host);

                for (int iinfo = 0; iinfo < infoLim; iinfo++)
                {
                    if (keyValues[iinfo].Length == 0)
                        continue;
                    ctx.SaveSubModel(string.Format(dirFormat, iinfo),
                        c => Save(ch, c, factory, ref keyValues[iinfo]));
                }
                ch.Done();
            }
        }
    }
}

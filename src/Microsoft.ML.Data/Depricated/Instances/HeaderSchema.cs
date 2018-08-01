// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma warning disable 420 // volatile with Interlocked.CompareExchange

using Float = System.Single;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Internal.Internallearn
{
    public abstract class FeatureNameCollection : IEnumerable<string>
    {
        private sealed class FeatureNameCollectionSchema : ISchema
        {
            private readonly VectorType _colType;
            private readonly VectorType _slotNamesType;

            private readonly FeatureNameCollection _collection;

            private readonly MetadataUtils.MetadataGetter<VBuffer<DvText>> _getSlotNames;

            public int ColumnCount => 1;

            public FeatureNameCollectionSchema(FeatureNameCollection collection)
            {
                Contracts.CheckValue(collection, nameof(collection));

                _collection = collection;
                _colType = new VectorType(NumberType.R4, collection.Count);
                _slotNamesType = new VectorType(TextType.Instance, collection.Count);
                _getSlotNames = GetSlotNames;
            }

            public string GetColumnName(int col)
            {
                Contracts.CheckParam(col == 0, nameof(col));
                return RoleMappedSchema.ColumnRole.Feature.Value;
            }

            public ColumnType GetColumnType(int col)
            {
                Contracts.CheckParam(col == 0, nameof(col));
                return _colType;
            }

            public void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                Contracts.CheckNonEmpty(kind, nameof(kind));
                Contracts.CheckParam(col == 0, nameof(col));

                if (kind == MetadataUtils.Kinds.SlotNames)
                    _getSlotNames.Marshal(col, ref value);
                else
                    throw MetadataUtils.ExceptGetMetadata();
            }

            public ColumnType GetMetadataTypeOrNull(string kind, int col)
            {
                Contracts.CheckNonEmpty(kind, nameof(kind));
                Contracts.CheckParam(col == 0, nameof(col));

                if (kind == MetadataUtils.Kinds.SlotNames)
                    return _slotNamesType;
                return null;
            }

            public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
            {
                Contracts.CheckParam(col == 0, nameof(col));
                yield return new KeyValuePair<string, ColumnType>(MetadataUtils.Kinds.SlotNames, _slotNamesType);
            }

            public bool TryGetColumnIndex(string name, out int col)
            {
                col = 0;
                return name == RoleMappedSchema.ColumnRole.Feature.Value;
            }

            private void GetSlotNames(int col, ref VBuffer<DvText> dst)
            {
                Contracts.Assert(col == 0);

                var nameList = new List<DvText>();
                var indexList = new List<int>();
                foreach (var kvp in _collection.GetNonDefaultFeatureNames())
                {
                    nameList.Add(new DvText(kvp.Value));
                    indexList.Add(kvp.Key);
                }

                var vals = dst.Values;
                if (Utils.Size(vals) < nameList.Count)
                    vals = new DvText[nameList.Count];
                Array.Copy(nameList.ToArray(), vals, nameList.Count);
                if (nameList.Count < _collection.Count)
                {
                    var indices = dst.Indices;
                    if (Utils.Size(indices) < indexList.Count)
                        indices = new int[indexList.Count];
                    Array.Copy(indexList.ToArray(), indices, indexList.Count);
                    dst = new VBuffer<DvText>(_collection.Count, nameList.Count, vals, indices);
                }
                else
                    dst = new VBuffer<DvText>(_collection.Count, vals, dst.Indices);
            }
        }

        private const string DefaultFmt = "f{0}";

        private volatile Dictionary<string, int> _lookup;
        private volatile object _lock;

        public abstract RoleMappedSchema Schema { get; }

        private FeatureNameCollection()
        {
        }

        public static FeatureNameCollection Create(string[] names)
        {
            return Create(Utils.Size(names), names);
        }

        public static FeatureNameCollection Create(int count, string[] names = null)
        {
            Contracts.CheckParam(count >= 0, nameof(count));
            Contracts.CheckValueOrNull(names);

            // See if we should use a sparse representation.
            int size = Math.Min(count, Utils.Size(names));
            if (size >= 30)
            {
                int cnn = names.Take(size).Count(x => x != null);
                if (cnn < size / 2)
                    return new Sparse(count, names, cnn);
            }

            return new Dense(count, names);
        }

        public static FeatureNameCollection Create(int count, Dictionary<int, string> map)
        {
            Contracts.CheckParam(count >= 0, nameof(count));
            Contracts.CheckValue(map, nameof(map));

            var items = map.Where(kvp => 0 <= kvp.Key && kvp.Key < count && kvp.Value != null);
            int lim = 0;
            int cnn = 0;
            foreach (var kvp in items)
            {
                if (lim <= kvp.Key)
                    lim = kvp.Key + 1;
                cnn++;
            }

            string[] names;
            if (lim >= 30 && cnn < lim / 2)
            {
                // Use sparse.
                var indices = new int[cnn];
                names = new string[cnn];
                int iv = 0;
                foreach (var kvp in items)
                {
                    indices[iv] = kvp.Key;
                    names[iv] = kvp.Value;
                    iv++;
                }
                Contracts.Assert(iv == cnn);
                return new Sparse(count, cnn, indices, names);
            }

            names = new string[lim];
            foreach (var kvp in items)
                names[kvp.Key] = kvp.Value;
            return new Dense(count, names);
        }

        public static FeatureNameCollection Create(RoleMappedSchema schema)
        {
            // REVIEW: This shim should be deleted as soon as is convenient.
            Contracts.CheckValue(schema, nameof(schema));
            Contracts.CheckParam(schema.Feature != null, nameof(schema), "Cannot create feature name collection if we have no features");
            Contracts.CheckParam(schema.Feature.Type.ValueCount > 0, nameof(schema), "Cannot create feature name collection if our features are not of known size");

            VBuffer<DvText> slotNames = default(VBuffer<DvText>);
            int len = schema.Feature.Type.ValueCount;
            if (schema.Schema.HasSlotNames(schema.Feature.Index, len))
                schema.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, schema.Feature.Index, ref slotNames);
            else
                slotNames = VBufferUtils.CreateEmpty<DvText>(len);
            string[] names = new string[slotNames.Count];
            for (int i = 0; i < slotNames.Count; ++i)
                names[i] = slotNames.Values[i].HasChars ? slotNames.Values[i].ToString() : null;
            if (slotNames.IsDense)
                return new Dense(names.Length, names);

            int[] indices = slotNames.Indices;
            if (indices == null)
                indices = new int[0];
            else if (indices.Length != slotNames.Count)
                Array.Resize(ref indices, slotNames.Count);
            return new Sparse(slotNames.Length, slotNames.Count, indices, names);
        }

        public const string LoaderSignature = "FeatureNamesExec";

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "FEATNAME",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: LoaderSignature);
        }

        public static void Save(ModelSaveContext ctx, ref VBuffer<DvText> names)
        {
            Contracts.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: number of features (size)
            // int: number of indices (-1 if dense)
            // int[]: indices (if sparse)
            // int[]: ids of names (matches either number of features or number of indices)

            ctx.Writer.Write(names.Length);
            if (names.IsDense)
            {
                ctx.Writer.Write(-1);
                for (int i = 0; i < names.Length; i++)
                    ctx.SaveStringOrNull(names.Values[i].ToString());
            }
            else
            {
                ctx.Writer.Write(names.Count);
                for (int ii = 0; ii < names.Count; ii++)
                    ctx.Writer.Write(names.Indices[ii]);
                for (int ii = 0; ii < names.Count; ii++)
                    ctx.SaveStringOrNull(names.Values[ii].ToString());
            }
        }

        public static FeatureNameCollection Create(ModelLoadContext ctx)
        {
            Contracts.AssertValue(ctx);
            ctx.CheckAtModel();
            ctx.CheckVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: number of features (size)
            // int: number of indices (0 if dense)
            // int[]: indices (if sparse)
            // int[]: ids of names (matches either number of features or number of indices
            var size = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(size >= 0);

            var isize = ctx.Reader.ReadInt32();
            Contracts.CheckDecode(isize >= -1);

            if (isize < 0)
            {
                // Dense case
                var names = new string[size];
                for (int i = 0; i < size; i++)
                {
                    var name = ctx.LoadStringOrNull();
                    names[i] = string.IsNullOrEmpty(name) ? null : name;
                }
                return Create(size, names);
            }
            var dict = new Dictionary<int, string>();
            var indices = new int[isize];
            var prev = -1;
            for (int ii = 0; ii < isize; ii++)
            {
                indices[ii] = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(prev < indices[ii]);
                prev = indices[ii];
            }
            Contracts.CheckDecode(prev < size);
            for (int ii = 0; ii < isize; ii++)
            {
                var name = ctx.LoadStringOrNull();
                if (!string.IsNullOrEmpty(name))
                    dict.Add(indices[ii], name);
            }
            return Create(size, dict);
        }

        public abstract int Count { get; }

        public abstract int NonDefaultCount { get; }

        public string this[int index] { get { return GetNameOrNull(index) ?? GetDefault(index); } }

        public abstract string GetNameOrNull(int index);

        public abstract IEnumerator<string> GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        private string GetDefault(int index)
        {
            Contracts.Assert(0 <= index & index < Count);
            return string.Format(DefaultFmt, index);
        }

        public bool TryLookup(string name, out int index)
        {
            if (_lookup == null)
                BuildLookup();

            if (_lookup.TryGetValue(name, out index))
                return true;

            // See if it is a default name.
            if (name.Length >= 2 && name[0] == 'f' && int.TryParse(name.Substring(1), out index) &&
                0 <= index && index < Count && name == this[index])
            {
                return true;
            }

            index = -1;
            return false;
        }

        private void BuildLookup()
        {
            if (_lock == null)
                Interlocked.CompareExchange(ref _lock, new object(), null);
            lock (_lock)
            {
                if (_lookup != null)
                    return;

                var lookup = new Dictionary<string, int>();
                PopulateLookup(lookup);
                _lookup = lookup;
            }
        }

        protected abstract void PopulateLookup(Dictionary<string, int> lookup);

        // Wrapper around an array of names. The array may be partial and may contain nulls.
        private sealed class Dense : FeatureNameCollection
        {
            private readonly int _count;
            private readonly string[] _names;

            public override RoleMappedSchema Schema { get; }

            public Dense(int count, string[] names)
            {
                Contracts.Assert(count >= 0);
                Contracts.AssertValueOrNull(names);

                _count = count;
                int size = Math.Min(Utils.Size(names), count);
                _names = new string[size];
                if (size > 0)
                    Array.Copy(names, _names, size);

                // REVIEW: This seems wrong. The default feature column name is "Features" yet the role is named "Feature".
                Schema = new RoleMappedSchema(new FeatureNameCollectionSchema(this),
                    roles: RoleMappedSchema.ColumnRole.Feature.Bind(RoleMappedSchema.ColumnRole.Feature.Value));
            }

            public override int Count => _count;

            public override int NonDefaultCount => _names.Length;

            public override string GetNameOrNull(int index)
            {
                Contracts.CheckParam(0 <= index && index < _count, nameof(index));
                return index < _names.Length ? _names[index] : null;
            }

            public override IEnumerator<string> GetEnumerator()
            {
                for (int i = 0; i < _names.Length; i++)
                    yield return _names[i] ?? GetDefault(i);
                for (int i = _names.Length; i < _count; i++)
                    yield return GetDefault(i);
            }

            protected override void PopulateLookup(Dictionary<string, int> lookup)
            {
                Contracts.AssertValue(lookup);

                // REVIEW: When there are dups, which one should win?
                for (int index = 0; index < _names.Length; index++)
                {
                    // REVIEW: Should we detect and report duplicates?
                    string name = _names[index];
                    if (name != null)
                        lookup[name] = index;
                }
            }

            protected override IEnumerable<KeyValuePair<int, string>> GetNonDefaultFeatureNames()
            {
                for (int i = 0; i < _names.Length; i++)
                    yield return new KeyValuePair<int, string>(i, _names[i]);
            }
        }

        protected abstract IEnumerable<KeyValuePair<int, string>> GetNonDefaultFeatureNames();

        // Wrapper around an array of names. The array may be partial and may contain nulls.
        private sealed class Sparse : FeatureNameCollection
        {
            // _length is the total number of features, and _count is the ones with a non-null name.
            private readonly int _length;
            private readonly int _count;
            private readonly string[] _names;
            private readonly int[] _indices;

            // This is the last position in _names/_indices accessed by the indexer. This is used
            // to speed up iterative access (avoid binary search on every access). Of course, it
            // is unlikely to help when multiple threads are iterating at the same time.
            private volatile int _ivPrev;

            private readonly RoleMappedSchema _schema;

            public override RoleMappedSchema Schema => _schema;

            /// <summary>
            /// This does NOT take ownership of the names array.
            /// </summary>
            public Sparse(int count, string[] names, int cnn)
            {
                Contracts.Assert(count >= 0);
                Contracts.AssertValue(names);

                _length = count;
                int size = Math.Min(names.Length, count);
                Contracts.Assert(size > 2 * cnn);

                _names = new string[cnn];
                _indices = new int[cnn];
                int cv = 0;
                for (int i = 0; i < size; i++)
                {
                    string name = names[i];
                    if (name != null)
                    {
                        Contracts.Assert(cv < cnn);
                        _names[cv] = name;
                        _indices[cv] = i;
                        cv++;
                    }
                }
                Contracts.Assert(cv == cnn);

                // REVIEW: This seems wrong. The default feature column name is "Features" yet the role is named "Feature".
                _schema = new RoleMappedSchema(new FeatureNameCollectionSchema(this),
                    roles: RoleMappedSchema.ColumnRole.Feature.Bind(RoleMappedSchema.ColumnRole.Feature.Value));
            }

            /// <summary>
            /// This takes ownership of the arrays.
            /// </summary>
            public Sparse(int length, int count, int[] indices, string[] names)
            {
                Contracts.Assert(count >= 0);
                Contracts.AssertValue(indices);
                Contracts.AssertValue(names);
                Contracts.Assert(indices.Length == names.Length);
                Contracts.Assert(indices.Length <= count);

                _length = length;
                _count = count;
                _indices = indices;
                _names = names;
            }

            public override int Count => _length;

            public override int NonDefaultCount => _count;

            public override string GetNameOrNull(int index)
            {
                Contracts.CheckParam(0 <= index && index < _length, nameof(index));

                // See if the cached _ivPrev helps.
                int iv = _ivPrev;
                if (iv < _indices.Length && _indices[iv] < index)
                {
                    if (++iv < _indices.Length && _indices[iv] < index)
                        iv = _indices.FindIndexSorted(iv + 1, _indices.Length, index);
                }
                else if (iv > 0 && _indices[iv - 1] >= index)
                {
                    if (--iv > 0 && _indices[iv - 1] >= index)
                        iv = _indices.FindIndexSorted(0, iv - 1, index);
                }
                Contracts.Assert(iv == _indices.FindIndexSorted(index));

                _ivPrev = iv;
                if (iv < _names.Length && _indices[iv] == index)
                    return _names[iv];

                return null;
            }

            public override IEnumerator<string> GetEnumerator()
            {
                int ii = 0;
                for (int i = 0; i < _count; i++)
                {
                    if (ii < _indices.Length && _indices[ii] == i)
                        yield return _names[ii++];
                    else
                        yield return GetDefault(i);
                }
            }

            protected override void PopulateLookup(Dictionary<string, int> lookup)
            {
                Contracts.AssertValue(lookup);

                // REVIEW: When there are dups, which one should win?
                for (int iv = 0; iv < _names.Length; iv++)
                {
                    // REVIEW: Should we detect and report duplicates?
                    string name = _names[iv];
                    Contracts.AssertValue(name);
                    lookup[name] = _indices[iv];
                }
            }

            protected override IEnumerable<KeyValuePair<int, string>> GetNonDefaultFeatureNames()
            {
                for (int i = 0; i < _indices.Length; i++)
                    yield return new KeyValuePair<int, string>(_indices[i], _names[i]);
            }
        }
    }
}

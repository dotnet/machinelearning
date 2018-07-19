// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Base class for handling the schema metadata API.
    /// </summary>
    public abstract class MetadataDispatcherBase
    {
        private bool _sealed;

        /// <summary>
        /// Information for a column.
        /// </summary>
        protected sealed class ColInfo
        {
            // The source schema to pass through metadata from. May be null, indicating none.
            public readonly ISchema SchemaSrc;
            // The source column index to pass through metadata from.
            public readonly int IndexSrc;
            // The metadata kind predicate indicating the kinds of metadata to pass through
            // from the source schema column. May be null, indicating all.
            public readonly Func<string, int, bool> FilterSrc;

            // The metadata getters.
            private readonly GetterInfo[] _getters;

            public int GetterCount { get { return _getters.Length; } }

            public IEnumerable<GetterInfo> Getters
            {
                get
                {
                    foreach (var g in _getters)
                        yield return g;
                }
            }

            public ColInfo(ISchema schemaSrc, int indexSrc, Func<string, int, bool> filterSrc,
                IEnumerable<GetterInfo> getters = null)
            {
                SchemaSrc = schemaSrc;
                IndexSrc = indexSrc;
                FilterSrc = filterSrc;
                _getters = getters != null ? getters.ToArray() : new GetterInfo[0];
            }

            public ColInfo UpdateGetters(IEnumerable<GetterInfo> getters)
            {
                if (getters == null)
                    return this;
                Contracts.CheckParam(!getters.Any(g => g == null), nameof(getters), "Invalid getter info");
                return new ColInfo(SchemaSrc, IndexSrc, FilterSrc, getters);
            }
        }

        /// <summary>
        /// Base class for metadata getters.
        /// </summary>
        protected abstract class GetterInfo
        {
            // The metadata kind.
            public readonly string Kind;
            // The metadata type.
            public readonly ColumnType Type;

            protected GetterInfo(string kind, ColumnType type)
            {
                Contracts.CheckNonWhiteSpace(kind, nameof(kind), "Invalid metadata kind");
                Contracts.CheckValue(type, nameof(type));
                Kind = kind;
                Type = type;
            }
        }

        /// <summary>
        /// Strongly typed base class for metadata getters. Introduces the abstract Get method.
        /// </summary>
        protected abstract class GetterInfo<TValue> : GetterInfo
        {
            protected GetterInfo(string kind, ColumnType type)
                : base(kind, type)
            {
            }

            public abstract void Get(int index, ref TValue value);
        }

        /// <summary>
        /// A delegate based metadata getter.
        /// </summary>
        protected sealed class GetterInfoDelegate<TValue> : GetterInfo<TValue>
        {
            public readonly MetadataUtils.MetadataGetter<TValue> Getter;

            public GetterInfoDelegate(string kind, ColumnType type, MetadataUtils.MetadataGetter<TValue> getter)
                : base(kind, type)
            {
                Contracts.Check(type.RawType == typeof(TValue), "Incompatible types");
                Contracts.CheckValue(getter, nameof(getter));
                Getter = getter;
            }

            public override void Get(int index, ref TValue value)
            {
                Getter(index, ref value);
            }
        }

        /// <summary>
        /// A primitive value based metadata getter.
        /// </summary>
        protected sealed class GetterInfoPrimitive<TValue> : GetterInfo<TValue>
        {
            // This is a MetadataGetter<TValue> where TValue is Type.RawType.
            public readonly TValue Value;

            public GetterInfoPrimitive(string kind, ColumnType type, TValue value)
                : base(kind, type)
            {
                Contracts.Check(type.RawType == typeof(TValue), "Incompatible types");
                Value = value;
            }

            public override void Get(int index, ref TValue value)
            {
                value = Value;
            }
        }

        private readonly ColInfo[] _infos;

        /// <summary>
        /// The number of columns.
        /// </summary>
        protected int ColCount { get { return _infos.Length; } }

        protected MetadataDispatcherBase(int colCount)
        {
            Contracts.CheckParam(colCount >= 0, nameof(colCount));
            _infos = new ColInfo[colCount];
        }

        /// <summary>
        /// Create a ColInfo with the indicated information and no GetterInfos. This doesn't
        /// register a column, only creates a ColInfo. Note that multiple columns can share
        /// the same ColInfo, if desired. Simply call RegisterColumn multiple times, passing
        /// the same ColInfo but different index values. This can only be called before Seal is called.
        /// </summary>
        protected ColInfo CreateInfo(ISchema schemaSrc = null, int indexSrc = -1,
            Func<string, int, bool> filterSrc = null)
        {
            Contracts.Check(!_sealed, "MetadataDispatcher sealed");
            Contracts.Check(schemaSrc == null || (0 <= indexSrc && indexSrc < schemaSrc.ColumnCount), "indexSrc out of range");
            Contracts.Check(filterSrc == null || schemaSrc != null, "filterSrc should be null if schemaSrc is null");
            return new ColInfo(schemaSrc, indexSrc, filterSrc);
        }

        /// <summary>
        /// Register the given ColInfo as the metadata handling information for the given
        /// column index. Throws if the given column index already has a ColInfo registered for it.
        /// This can only be called before Seal is called.
        /// </summary>
        protected void RegisterColumn(int index, ColInfo info)
        {
            Contracts.Check(!_sealed, "MetadataDispatcher sealed");
            Contracts.CheckValue(info, nameof(info));
            Contracts.CheckParam(0 <= index && index < _infos.Length, nameof(index), "Out of range");
            Contracts.CheckParam(_infos[index] == null, nameof(index), "Column already registered");
            _infos[index] = info;
        }

        /// <summary>
        /// Seals this dispatcher from further column registrations. This must be called before any
        /// metadata methods are called, otherwise an exception is thrown.
        /// </summary>
        protected void Seal()
        {
            _sealed = true;
        }

        /// <summary>
        /// Returns the ColInfo registered for the given column index, if there is one. This may be called
        /// before or after Seal is called.
        /// </summary>
        protected ColInfo GetColInfoOrNull(int index)
        {
            Contracts.CheckParam(0 <= index && index < _infos.Length, nameof(index));
            return _infos[index];
        }

        /// <summary>
        /// Gets the metadata kinds and types for the given column index.
        /// This can only be called after Seal is called.
        /// </summary>
        public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int index)
        {
            Contracts.Check(_sealed, "MetadataDispatcher not sealed");

            var info = GetColInfoOrNull(index);
            if (info == null)
                return Enumerable.Empty<KeyValuePair<string, ColumnType>>();
            return GetTypesCore(index, info);
        }

        private IEnumerable<KeyValuePair<string, ColumnType>> GetTypesCore(int index, ColInfo info)
        {
            Contracts.Assert(_sealed);
            Contracts.AssertValue(info);

            HashSet<string> kinds = null;
            if (info.GetterCount > 0)
            {
                if (info.SchemaSrc != null)
                    kinds = new HashSet<string>();
                foreach (var g in info.Getters)
                {
                    yield return new KeyValuePair<string, ColumnType>(g.Kind, g.Type);
                    if (kinds != null)
                        kinds.Add(g.Kind);
                }
            }

            if (info.SchemaSrc == null)
                yield break;

            // Pass through from base, with filtering.
            foreach (var kvp in info.SchemaSrc.GetMetadataTypes(info.IndexSrc))
            {
                if (kinds != null && kinds.Contains(kvp.Key))
                    continue;
                if (info.FilterSrc != null && !info.FilterSrc(kvp.Key, index))
                    continue;
                yield return kvp;
            }
        }

        /// <summary>
        /// Gets the metadata type for the given metadata kind and column index, if there is one.
        /// This can only be called after Seal is called.
        /// </summary>
        public ColumnType GetMetadataTypeOrNull(string kind, int index)
        {
            Contracts.Check(_sealed, "MetadataDispatcher not sealed");

            var info = GetColInfoOrNull(index);
            if (info == null)
                return null;

            foreach (var g in info.Getters)
            {
                if (g.Kind == kind)
                    return g.Type;
            }

            if (info.SchemaSrc == null)
                return null;
            if (info.FilterSrc != null && !info.FilterSrc(kind, index))
                return null;
            return info.SchemaSrc.GetMetadataTypeOrNull(kind, info.IndexSrc);
        }

        /// <summary>
        /// Gets the metadata for the given metadata kind and column index. Throws if there isn't any.
        /// This can only be called after Seal is called.
        /// </summary>
        public void GetMetadata<TValue>(IExceptionContext ectx, string kind, int index, ref TValue value)
        {
            ectx.Check(_sealed, "MetadataDispatcher not sealed");
            ectx.Check(0 <= index && index < _infos.Length);

            var info = _infos[index];
            if (info == null)
                throw ectx.ExceptGetMetadata();

            foreach (var g in info.Getters)
            {
                if (g.Kind == kind)
                {
                    var getter = g as GetterInfo<TValue>;
                    if (getter == null)
                        throw ectx.ExceptGetMetadata();
                    getter.Get(index, ref value);
                    return;
                }
            }

            if (info.SchemaSrc == null || info.FilterSrc != null && !info.FilterSrc(kind, index))
                throw ectx.ExceptGetMetadata();
            info.SchemaSrc.GetMetadata(kind, info.IndexSrc, ref value);
        }
    }

    /// <summary>
    /// For handling the schema metadata API. Call one of the BuildMetadata methods to get
    /// a builder for a particular column. Wrap the return in a using statement. Disposing the builder
    /// records the metadata for the column. Call Seal() once all metadata is constructed.
    /// </summary>
    public sealed class MetadataDispatcher : MetadataDispatcherBase
    {
        public MetadataDispatcher(int colCount)
            : base(colCount)
        {
        }

        /// <summary>
        /// Start building metadata for a column that doesn't pass through any metadata from
        /// a source column.
        /// </summary>
        public Builder BuildMetadata(int index)
        {
            return new Builder(this, index);
        }

        /// <summary>
        /// Start building metadata for a column that passes through all metadata from
        /// a source column.
        /// </summary>
        public Builder BuildMetadata(int index, ISchema schemaSrc, int indexSrc)
        {
            Contracts.CheckValue(schemaSrc, nameof(schemaSrc));
            return new Builder(this, index, schemaSrc, indexSrc);
        }

        /// <summary>
        /// Start building metadata for a column that passes through metadata of certain kinds from
        /// a source column. The kinds that are passed through are those for which
        /// <paramref name="filterSrc"/> returns true.
        /// </summary>
        public Builder BuildMetadata(int index, ISchema schemaSrc, int indexSrc, Func<string, int, bool> filterSrc)
        {
            Contracts.CheckValue(schemaSrc, nameof(schemaSrc));
            return new Builder(this, index, schemaSrc, indexSrc, filterSrc);
        }

        /// <summary>
        /// Start building metadata for a column that passes through metadata of the given kind from
        /// a source column.
        /// </summary>
        public Builder BuildMetadata(int index, ISchema schemaSrc, int indexSrc, string kindSrc)
        {
            Contracts.CheckValue(schemaSrc, nameof(schemaSrc));
            Contracts.CheckNonWhiteSpace(kindSrc, nameof(kindSrc));
            return new Builder(this, index, schemaSrc, indexSrc, (k, i) => k == kindSrc);
        }

        /// <summary>
        /// Start building metadata for a column that passes through metadata of the given kinds from
        /// a source column.
        /// </summary>
        public Builder BuildMetadata(int index, ISchema schemaSrc, int indexSrc, params string[] kindsSrc)
        {
            Contracts.CheckValue(schemaSrc, nameof(schemaSrc));
            Contracts.CheckParam(Utils.Size(kindsSrc) >= 2, nameof(kindsSrc));
            Contracts.CheckParam(!kindsSrc.Any(k => string.IsNullOrWhiteSpace(k)), nameof(kindsSrc));

            var set = new HashSet<string>(kindsSrc);
            return new Builder(this, index, schemaSrc, indexSrc, (k, i) => set.Contains(k));
        }

        new public void Seal()
        {
            base.Seal();
        }

        /// <summary>
        /// The builder for metadata for a particular column.
        /// </summary>
        public sealed class Builder : IDisposable
        {
            private readonly int _index;
            private MetadataDispatcher _md;
            private ColInfo _info;
            private List<GetterInfo> _getters;

            /// <summary>
            /// This should really be private to MetadataDispatcher, but C#'s accessibility model doesn't
            /// allow restricting to an outer class.
            /// </summary>
            internal Builder(MetadataDispatcher md, int index,
                ISchema schemaSrc = null, int indexSrc = -1, Func<string, int, bool> filterSrc = null)
            {
                Contracts.CheckValue(md, nameof(md));
                Contracts.CheckParam(0 <= index && index < md.ColCount, nameof(index));

                _index = index;
                _md = md;
                _info = _md.CreateInfo(schemaSrc, indexSrc, filterSrc);

                var tmp = _md.GetColInfoOrNull(_index);
                Contracts.Check(tmp == null, "Duplicate building of metadata");
            }

            /// <summary>
            /// Add metadata of the given kind. When requested, the metadata is fetched by calling the given delegate.
            /// </summary>
            public void AddGetter<TValue>(string kind, ColumnType type,
                MetadataUtils.MetadataGetter<TValue> getter)
            {
                Contracts.Check(_md != null, "Builder disposed");
                Contracts.CheckNonEmpty(kind, nameof(kind));
                Contracts.CheckValue(type, nameof(type));
                Contracts.CheckValue(getter, nameof(getter));
                Contracts.CheckParam(type.RawType == typeof(TValue), nameof(type), "Given type doesn't match type parameter");

                if (_getters != null && _getters.Any(g => g.Kind == kind))
                    throw Contracts.Except("Duplicate specification of metadata");
                Utils.Add(ref _getters, new GetterInfoDelegate<TValue>(kind, type, getter));
            }

            /// <summary>
            /// Add metadata of the given kind, with the given value.
            /// </summary>
            public void AddPrimitive<TValue>(string kind, ColumnType type, TValue value)
            {
                Contracts.Check(_md != null, "Builder disposed");
                Contracts.CheckNonEmpty(kind, nameof(kind));
                Contracts.CheckValue(type, nameof(type));
                Contracts.CheckParam(type.RawType == typeof(TValue), nameof(type), "Given type doesn't match type parameter");
                Contracts.CheckParam(type.IsPrimitive, nameof(type), "Must be a primitive type");

                if (_getters != null && _getters.Any(g => g.Kind == kind))
                    throw Contracts.Except("Duplicate specification of metadata");
                Utils.Add(ref _getters, new GetterInfoPrimitive<TValue>(kind, type, value));
            }

            /// <summary>
            /// Close out the builder. This registers the metadata with the dispatcher.
            /// </summary>
            public void Dispose()
            {
                if (_md == null)
                    return;

                Contracts.Assert(_info != null);

                var md = _md;
                _md = null;
                var info = _info;
                _info = null;
                var getters = _getters;
                _getters = null;

                if (Utils.Size(getters) > 0)
                    info = info.UpdateGetters(getters);

                md.RegisterColumn(_index, info);
            }
        }
    }
}

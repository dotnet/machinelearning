// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Transforms.Categorical;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace Microsoft.ML.Runtime.Data
{
    public abstract class SourceNameColumnBase
    {
        [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the new column", ShortName = "name")]
        public string Name;

        [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the source column", ShortName = "src")]
        public string Source;

        /// <summary>
        /// For parsing from a string. This supports "name" and "name:source".
        /// Derived classes that want to provide parsing functionality to the CmdParser need to implement
        /// a static Parse method. That method can call this (directly or indirectly) to handle the supported
        /// syntax.
        /// </summary>
        protected virtual bool TryParse(string str)
        {
            Contracts.AssertNonEmpty(str);
            return ColumnParsingUtils.TryParse(str, out Name, out Source);
        }

        /// <summary>
        /// For parsing from a string. This supports "name" and "name:source" and "name:extra:source". For the last
        /// form, the out extra parameter is sort accordingly. For the other forms, extra is set to null.
        /// Derived classes that want to provide parsing functionality to the CmdParser need to implement
        /// a static Parse method. That method can call this (directly or indirectly) to handle the supported
        /// syntax.
        /// </summary>
        protected bool TryParse(string str, out string extra)
        {
            Contracts.AssertNonEmpty(str);
            return ColumnParsingUtils.TryParse(str, out Name, out Source, out extra);
        }

        /// <summary>
        /// The core unparsing functionality, for generating succinct command line forms "name" and "name:source".
        /// </summary>
        protected virtual bool TryUnparseCore(StringBuilder sb)
        {
            Contracts.AssertValue(sb);

            if (!TrySanitize())
                return false;
            if (CmdQuoter.NeedsQuoting(Name) || CmdQuoter.NeedsQuoting(Source))
                return false;

            sb.Append(Name);
            if (Source != Name)
                sb.Append(':').Append(Source);
            return true;
        }

        /// <summary>
        /// The core unparsing functionality, for generating the succinct command line form "name:extra:source".
        /// </summary>
        protected virtual bool TryUnparseCore(StringBuilder sb, string extra)
        {
            Contracts.AssertValue(sb);
            Contracts.AssertNonEmpty(extra);

            if (!TrySanitize())
                return false;
            if (CmdQuoter.NeedsQuoting(Name) || CmdQuoter.NeedsQuoting(Source))
                return false;

            sb.Append(Name).Append(':').Append(extra).Append(':').Append(Source);
            return true;
        }

        /// <summary>
        /// If both of name and source are null or white-space, return false.
        /// Otherwise, if one is null or white-space, assign that one the other's value.
        /// </summary>
        public bool TrySanitize()
        {
            if (string.IsNullOrWhiteSpace(Name))
                Name = Source;
            else if (string.IsNullOrWhiteSpace(Source))
                Source = Name;
            return !string.IsNullOrWhiteSpace(Name);
        }
    }

    public abstract class OneToOneColumn : SourceNameColumnBase
    {
    }

    public interface IOneToOneColumn
    {
        string Name { get; set; }
        string Source { get; set; }
    }

    public abstract class OneToOneColumn<T>
        where T : IOneToOneColumn, new()
    {
        public static T Create(string source)
        {
            return new T() { Name = source, Source = source };
        }

        public static T Create(string name, string source)
        {
            return new T() { Name = name, Source = source };
        }
    }

    public abstract class ManyToOneColumn
    {
        [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the new column", ShortName = "name")]
        public string Name;

        [Argument(ArgumentType.Multiple, HelpText = "Name of the source column", ShortName = "src")]
        public string[] Source;

        /// <summary>
        /// The parsing functionality for custom parsing from a string. This supports "name" and "name:sources",
        /// where sources is a comma separated list of source column names.
        /// </summary>
        protected virtual bool TryParse(string str)
        {
            Contracts.AssertNonEmpty(str);

            int ich = str.IndexOf(':');
            if (ich < 0 && str.IndexOf(',') < 0)
            {
                Name = str;
                Source = new string[] { str };
                return true;
            }

            if (0 < ich && ich < str.Length - 1)
            {
                Name = str.Substring(0, ich);
                var src = str.Substring(ich + 1);
                if (src.Contains(":"))
                    return false;
                Source = src.Split(',');
                return Source.All(s => !string.IsNullOrEmpty(s));
            }

            return false;
        }

        /// <summary>
        /// Parsing functionality for custom parsing from a string with an "extra" value between name and sources.
        /// This supports "name", "name:sources" and "name:extra:sources".
        /// </summary>
        protected bool TryParse(string str, out string extra)
        {
            Contracts.AssertNonEmpty(str);

            extra = null;
            int ich = str.IndexOf(':');
            if (ich < 0)
            {
                Name = str;
                Source = new string[] { str };
                return true;
            }
            if (ich == 0 || ich >= str.Length - 1)
                return false;

            Name = str.Substring(0, ich);

            int ichMin = ich + 1;
            ich = str.IndexOf(':', ichMin);
            string src;
            if (ich < 0)
                src = str.Substring(ichMin);
            else if (ich == ichMin || ich >= str.Length - 1)
                return false;
            else
            {
                extra = str.Substring(ichMin, ich - ichMin);
                src = str.Substring(ich + 1);
                if (src.Contains(':'))
                    return false;
            }
            Source = src.Split(',');
            return Source.All(s => !string.IsNullOrEmpty(s));
        }

        protected virtual bool TryUnparseCore(StringBuilder sb)
        {
            Contracts.AssertValue(sb);

            if (string.IsNullOrWhiteSpace(Name) || Utils.Size(Source) == 0)
                return false;
            if (CmdQuoter.NeedsQuoting(Name))
                return false;
            if (Source.Any(x => CmdQuoter.NeedsQuoting(x) || x.Contains(",")))
                return false;
            if (Source.Length == 1 && Source[0] == Name)
            {
                sb.Append(Name);
                return true;
            }
            sb.Append(Name).Append(':');
            string pre = "";
            foreach (var src in Source)
            {
                sb.Append(pre).Append(src);
                pre = ",";
            }
            return true;
        }

        protected virtual bool TryUnparseCore(StringBuilder sb, string extra)
        {
            Contracts.AssertNonEmpty(extra);

            if (string.IsNullOrWhiteSpace(Name) || Utils.Size(Source) == 0)
                return false;
            if (CmdQuoter.NeedsQuoting(Name))
                return false;
            if (Source.Any(x => CmdQuoter.NeedsQuoting(x) || x.Contains(",")))
                return false;
            sb.Append(Name).Append(':').Append(extra).Append(':');
            string pre = "";
            foreach (var src in Source)
            {
                sb.Append(pre).Append(src);
                pre = ",";
            }
            return true;
        }
    }

    public interface IManyToOneColumn
    {
        string Name { get; set; }
        string[] Source { get; set; }
    }

    public abstract class ManyToOneColumn<T>
        where T : IManyToOneColumn, new()
    {
        public static T Create(string name, params string[] source)
        {
            return new T() { Name = name, Source = source };
        }
    }

    /// <summary>
    /// Base class that abstracts passing input columns through (with possibly different indices) and adding
    /// InfoCount additional columns. If an added column has the same name as a non-hidden input column, it hides
    /// the input column, and is placed immediately after the input column. Otherwise, the added column is placed
    /// at the end. By default, newly added columns have no metadata (but this can be overriden).
    /// </summary>
    public abstract class ColumnBindingsBase : ISchema
    {
        public readonly ISchema Input;

        // Mapping from name to index into Infos for the columns that we "generate".
        // Some of these might "hide" input columns.
        private readonly Dictionary<string, int> _nameToInfoIndex;

        // The names of the newly added columns.
        private readonly string[] _names;

        // This maps from index into Infos to the corresponding output column index.
        private readonly int[] _mapIinfoToCol;

        // Mapping from output column index to either input column index or the bitwise complement of the
        // index into Infos (for the columns that we "generate").
        private readonly int[] _colMap;

        // Conversion to the eager schema.
        private readonly Lazy<Schema> _convertedSchema;

        public Schema AsSchema => _convertedSchema.Value;

        /// <summary>
        /// Constructor that takes an input schema and adds no new columns.
        /// This is utilized by lambda transforms if the output happens to have no columns.
        /// </summary>
        /// <param name="input"></param>
        protected ColumnBindingsBase(ISchema input)
        {
            Contracts.CheckValue(input, nameof(input));

            Input = input;

            _names = new string[0];
            _nameToInfoIndex = new Dictionary<string, int>();
            Contracts.Assert(_nameToInfoIndex.Count == _names.Length);
            ComputeColumnMapping(Input, _names, out _colMap, out _mapIinfoToCol);
            _convertedSchema = new Lazy<Schema>(() => Schema.Create(this), LazyThreadSafetyMode.PublicationOnly);
        }

        /// <summary>
        /// Constructor taking the input schema and new column names. Names must be non-empty and
        /// each name must be non-white-space. The names must be unique but can match existing names
        /// in schemaInput. For error reporting, this assumes that the names come from a user-supplied
        /// parameter named "column". This takes ownership of the params array of names.
        /// </summary>
        protected ColumnBindingsBase(ISchema input, bool user, params string[] names)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckNonEmpty(names, nameof(names));

            Input = input;
            _names = names;
            _nameToInfoIndex = new Dictionary<string, int>(names.Length);

            // In lieu of actual protections, I have the following silly asserts, so we can have some
            // warning if we decide to rename this argument, and so know to change the below hard-coded
            // standard column name.
            const string standardColumnArgName = "Column";
            Contracts.Assert(nameof(TermTransform.Arguments.Column) == standardColumnArgName);
            Contracts.Assert(nameof(ConcatTransform.Arguments.Column) == standardColumnArgName);

            for (int iinfo = 0; iinfo < names.Length; iinfo++)
            {
                var name = names[iinfo];
                if (string.IsNullOrWhiteSpace(name))
                {
                    throw user ?
#pragma warning disable MSML_ContractsNameUsesNameof // Unfortunately, there is no base class for the columns bindings.
                        Contracts.ExceptUserArg(standardColumnArgName, "New column needs a name") :
#pragma warning restore MSML_ContractsNameUsesNameof
                        Contracts.ExceptDecode("New column needs a name");
                }
                if (_nameToInfoIndex.ContainsKey(name))
                {
                    throw user ?
#pragma warning disable MSML_ContractsNameUsesNameof // Unfortunately, there is no base class for the columns bindings.
                        Contracts.ExceptUserArg(standardColumnArgName, "New column '{0}' specified multiple times", name) :
#pragma warning restore MSML_ContractsNameUsesNameof
                        Contracts.ExceptDecode("New column '{0}' specified multiple times", name);
                }
                _nameToInfoIndex.Add(name, iinfo);
            }
            Contracts.Assert(_nameToInfoIndex.Count == names.Length);

            ComputeColumnMapping(Input, names, out _colMap, out _mapIinfoToCol);
            _convertedSchema = new Lazy<Schema>(() => Schema.Create(this), LazyThreadSafetyMode.PublicationOnly);
        }

        private static void ComputeColumnMapping(ISchema input, string[] names, out int[] colMap, out int[] mapIinfoToCol)
        {
            // To compute the column mapping information, first populate:
            // * _colMap[src] with the ~ of the iinfo that hides src (zero for none).
            // * _mapIinfoToCol[iinfo] with the ~ of the source column that iinfo hides (zero for none).
            colMap = new int[input.ColumnCount + names.Length];
            mapIinfoToCol = new int[names.Length];
            for (int iinfo = 0; iinfo < names.Length; iinfo++)
            {
                var name = names[iinfo];
                int colHidden;
                if (input.TryGetColumnIndex(name, out colHidden))
                {
                    Contracts.Check(0 <= colHidden && colHidden < input.ColumnCount);
                    var str = input.GetColumnName(colHidden);
                    Contracts.Check(str == name);
                    Contracts.Check(colMap[colHidden] == 0);
                    mapIinfoToCol[iinfo] = ~colHidden;
                    colMap[colHidden] = ~iinfo;
                }
            }

            // Now back-fill the column mapping.
            int colDst = colMap.Length;
            for (int iinfo = names.Length; --iinfo >= 0;)
            {
                Contracts.Assert(mapIinfoToCol[iinfo] <= 0);
                if (mapIinfoToCol[iinfo] == 0)
                {
                    colMap[--colDst] = ~iinfo;
                    mapIinfoToCol[iinfo] = colDst;
                }
            }
            for (int colSrc = input.ColumnCount; --colSrc >= 0;)
            {
                Contracts.Assert(colMap[colSrc] <= 0);
                if (colMap[colSrc] < 0)
                {
                    Contracts.Assert(colDst > 1);
                    int iinfo = ~colMap[colSrc];
                    Contracts.Assert(0 <= iinfo && iinfo < names.Length);
                    Contracts.Assert(mapIinfoToCol[iinfo] == ~colSrc);
                    colMap[--colDst] = ~iinfo;
                    mapIinfoToCol[iinfo] = colDst;
                }
                Contracts.Assert(colDst > 0);
                colMap[--colDst] = colSrc;
            }
            Contracts.Assert(colDst == 0);
        }

        public int ColumnCount => _colMap.Length;

        // REVIEW: Ideally this wouldn't be public, but typically cursors want access to it.

        /// <summary>
        /// The number of added columns.
        /// </summary>
        public int InfoCount => _mapIinfoToCol.Length;

        // REVIEW: Ideally this wouldn't be public, but typically cursors want access to it.

        /// <summary>
        /// This maps a column index for this schema to either a source column index (when
        /// <paramref name="isSrcColumn"/> is true), or to an "iinfo" index of an added column
        /// (when <paramref name="isSrcColumn"/> is false).
        /// </summary>
        /// <param name="isSrcColumn">Whether the return index is for a source column</param>
        /// <param name="col">The column index for this schema</param>
        /// <returns>The index (either source index or iinfo index)</returns>
        public int MapColumnIndex(out bool isSrcColumn, int col)
        {
            Contracts.Assert(0 <= col && col < _colMap.Length);
            int index = _colMap[col];
            if (index < 0)
            {
                index = ~index;
                Contracts.Assert(0 <= index && index < InfoCount);
                isSrcColumn = false;
            }
            else
            {
                Contracts.Assert(index < Input.ColumnCount);
                isSrcColumn = true;
            }
            return index;
        }

        // REVIEW: Ideally this wouldn't be public, but typically cursors want access to it.

        /// <summary>
        /// This maps from an index to an added column "info" to a column index.
        /// </summary>
        public int MapIinfoToCol(int iinfo)
        {
            Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
            return _mapIinfoToCol[iinfo];
        }

        public bool TryGetColumnIndex(string name, out int col)
        {
            Contracts.CheckValueOrNull(name);
            if (name == null)
            {
                col = default(int);
                return false;
            }

            int iinfo;
            if (TryGetColumnIndexCore(name, out iinfo))
            {
                Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
                col = MapIinfoToCol(iinfo);
                return true;
            }

            // REVIEW: Should we keep a dictionary for this mapping? This first looks up
            // the source column index, then does a linear scan in _colMap, starting at the src
            // slot (since source columns can only shift to larger indices).
            int src;
            if (Input.TryGetColumnIndex(name, out src))
            {
                Contracts.Assert(0 <= src && src < Input.ColumnCount);
                int res = src;
                for (; ; res++)
                {
                    Contracts.Assert(0 <= res && res < ColumnCount);
                    Contracts.Assert(_colMap[res] <= src);
                    if (_colMap[res] == src)
                    {
                        col = res;
                        return true;
                    }
                }
            }

            col = default(int);
            return false;
        }

        public string GetColumnName(int col)
        {
            Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));

            bool isSrc;
            int index = MapColumnIndex(out isSrc, col);
            if (isSrc)
                return Input.GetColumnName(index);
            return GetColumnNameCore(index);
        }

        public ColumnType GetColumnType(int col)
        {
            Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));

            bool isSrc;
            int index = MapColumnIndex(out isSrc, col);
            if (isSrc)
                return Input.GetColumnType(index);
            return GetColumnTypeCore(index);
        }

        public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
        {
            Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));

            bool isSrc;
            int index = MapColumnIndex(out isSrc, col);
            if (isSrc)
                return Input.GetMetadataTypes(index);
            Contracts.Assert(0 <= index && index < InfoCount);
            return GetMetadataTypesCore(index);
        }

        public ColumnType GetMetadataTypeOrNull(string kind, int col)
        {
            Contracts.CheckNonEmpty(kind, nameof(kind));
            Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));

            bool isSrc;
            int index = MapColumnIndex(out isSrc, col);
            if (isSrc)
                return Input.GetMetadataTypeOrNull(kind, index);
            Contracts.Assert(0 <= index && index < InfoCount);
            return GetMetadataTypeCore(kind, index);
        }

        public void GetMetadata<TValue>(string kind, int col, ref TValue value)
        {
            Contracts.CheckNonEmpty(kind, nameof(kind));
            Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));

            bool isSrc;
            int index = MapColumnIndex(out isSrc, col);
            if (isSrc)
                Input.GetMetadata(kind, index, ref value);
            else
            {
                Contracts.Assert(0 <= index && index < InfoCount);
                GetMetadataCore(kind, index, ref value);
            }
        }

        protected bool TryGetColumnIndexCore(string name, out int iinfo)
        {
            Contracts.AssertValue(name);
            return _nameToInfoIndex.TryGetValue(name, out iinfo);
        }

        protected string GetColumnNameCore(int iinfo)
        {
            Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
            return _names[iinfo];
        }

        protected abstract ColumnType GetColumnTypeCore(int iinfo);

        protected virtual IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypesCore(int iinfo)
        {
            Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
            return Enumerable.Empty<KeyValuePair<string, ColumnType>>();
        }

        protected virtual ColumnType GetMetadataTypeCore(string kind, int iinfo)
        {
            Contracts.AssertNonEmpty(kind);
            Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
            return null;
        }

        protected virtual void GetMetadataCore<TValue>(string kind, int iinfo, ref TValue value)
        {
            Contracts.AssertNonEmpty(kind);
            Contracts.Assert(0 <= iinfo && iinfo < InfoCount);
            throw MetadataUtils.ExceptGetMetadata();
        }

        /// <summary>
        /// The given predicate maps from output column index to whether the column is active.
        /// This builds an array of bools of length ColumnCount containing the results of calling
        /// predicate on each column index.
        /// </summary>
        public bool[] GetActive(Func<int, bool> predicate)
        {
            return Utils.BuildArray(ColumnCount, predicate);
        }

        /// <summary>
        /// The given predicate maps from output column index to whether the column is active.
        /// This builds an array of bools of length Input.ColumnCount containing the results of calling
        /// predicate on the output column index corresponding to each input column index.
        /// </summary>
        public bool[] GetActiveInput(Func<int, bool> predicate)
        {
            Contracts.AssertValue(predicate);

            var active = new bool[Input.ColumnCount];
            for (int dst = 0; dst < _colMap.Length; dst++)
            {
                int src = _colMap[dst];
                Contracts.Assert(-InfoCount <= src && src < Input.ColumnCount);
                if (src >= 0 && predicate(dst))
                    active[src] = true;
            }
            return active;
        }

        /// <summary>
        /// Determine whether any columns generated by this transform are active.
        /// </summary>
        public bool AnyNewColumnsActive(Func<int, bool> predicate)
        {
            Contracts.AssertValue(predicate);

            foreach (int col in _mapIinfoToCol)
            {
                if (predicate(col))
                    return true;
            }
            return false;
        }
    }

    /// <summary>
    /// Class that encapsulates passing input columns through (with possibly different indices) and adding
    /// additional columns. If an added column has the same name as a non-hidden input column, it hides
    /// the input column, and is placed immediately after the input column. Otherwise, the added column is placed
    /// at the end.
    /// This class is intended to simplify predicate propagation for this case.
    /// </summary>
    public sealed class ColumnBindings
    {
        // Indices of columns in the merged schema. Old indices are as is, new indices are stored as ~idx.
        private readonly int[] _colMap;

        /// <summary>
        /// The indices of added columns in the <see cref="Schema"/>.
        /// </summary>
        public IReadOnlyList<int> AddedColumnIndices { get; }

        /// <summary>
        /// The input schema.
        /// </summary>
        public Schema InputSchema { get; }

        /// <summary>
        /// The merged schema.
        /// </summary>
        public Schema Schema { get; }

        /// <summary>
        /// Create a new instance of <see cref="ColumnBindings"/>.
        /// </summary>
        /// <param name="input">The input schema that we're adding columns to.</param>
        /// <param name="addedColumns">The columns being added.</param>
        public ColumnBindings(Schema input, Schema.Column[] addedColumns)
        {
            Contracts.CheckValue(input, nameof(input));
            Contracts.CheckValue(addedColumns, nameof(addedColumns));

            InputSchema = input;

            // Construct the indices.
            var indices = new List<int>();
            var namesUsed = new HashSet<string>();
            for (int i = 0; i < input.ColumnCount; i++)
            {
                namesUsed.Add(input[i].Name);
                indices.Add(i);
            }

            for (int i = 0; i < addedColumns.Length; i++)
            {
                string name = addedColumns[i].Name;
                if (namesUsed.Add(name))
                {
                    // New name. Append to the end.
                    indices.Add(~i);
                }
                else
                {
                    // Old name. Find last instance and add after it.
                    for (int j = indices.Count - 1; j >= 0; j--)
                    {
                        var colName = indices[j] >= 0 ? input[indices[j]].Name : addedColumns[~indices[j]].Name;
                        if (colName == name)
                        {
                            indices.Insert(j + 1, ~i);
                            break;
                        }
                    }
                }
            }
            Contracts.Assert(indices.Count == addedColumns.Length + input.ColumnCount);

            // Create the output schema.
            var schemaColumns = indices.Select(idx => idx >= 0 ? input[idx] : addedColumns[~idx]);
            Schema = new Schema(schemaColumns);

            // Memorize column maps.
            _colMap = indices.ToArray();
            var addedIndices = new int[addedColumns.Length];
            for (int i = 0; i < _colMap.Length; i++)
            {
                int colIndex = _colMap[i];
                if (colIndex < 0)
                {
                    Contracts.Assert(addedIndices[~colIndex] == 0);
                    addedIndices[~colIndex] = i;
                }
            }

            AddedColumnIndices = addedIndices.AsReadOnly();
        }

        /// <summary>
        /// This maps a column index for this schema to either a source column index (when
        /// <paramref name="isSrcColumn"/> is true), or to an "iinfo" index of an added column
        /// (when <paramref name="isSrcColumn"/> is false).
        /// </summary>
        /// <param name="isSrcColumn">Whether the return index is for a source column</param>
        /// <param name="col">The column index for this schema</param>
        /// <returns>The index (either source index or iinfo index)</returns>
        public int MapColumnIndex(out bool isSrcColumn, int col)
        {
            Contracts.Assert(0 <= col && col < _colMap.Length);
            int index = _colMap[col];
            if (index < 0)
            {
                index = ~index;
                Contracts.Assert(index < AddedColumnIndices.Count);
                isSrcColumn = false;
            }
            else
            {
                Contracts.Assert(index < InputSchema.ColumnCount);
                isSrcColumn = true;
            }
            return index;
        }

        /// <summary>
        /// The given predicate maps from output column index to whether the column is active.
        /// This builds an array of bools of length Input.ColumnCount containing the results of calling
        /// predicate on the output column index corresponding to each input column index.
        /// </summary>
        public bool[] GetActiveInput(Func<int, bool> predicate)
        {
            Contracts.AssertValue(predicate);

            var active = new bool[InputSchema.ColumnCount];
            for (int dst = 0; dst < _colMap.Length; dst++)
            {
                int src = _colMap[dst];
                Contracts.Assert(-AddedColumnIndices.Count <= src && src < InputSchema.ColumnCount);
                if (src >= 0 && predicate(dst))
                    active[src] = true;
            }
            return active;
        }
    }

    /// <summary>
    /// Base type for bindings with multiple new columns, each mapping from multiple source columns.
    /// The column strings are parsed as D:S where D is the name of the new column and S is a comma separated
    /// list of source columns. A column string S with no colon is interpreted as S:S, so the destination
    /// column has the same name as the source column. This form requires S to not contain commas.
    /// Note that this base type requires columns of type typeMulti to have known size.
    /// </summary>
    public abstract class ManyToOneColumnBindingsBase : ColumnBindingsBase
    {
        public sealed class ColInfo
        {
            // Total size of all source columns, zero for variable (non-vector columns contribute 1).
            public readonly int SrcSize;
            // Indices of the source columns
            public readonly int[] SrcIndices;
            // Types of the source columns.
            public readonly ColumnType[] SrcTypes;

            public ColInfo(int srcSize, int[] srcIndices, ColumnType[] srcTypes)
            {
                Contracts.Assert(srcSize >= 0); // Zero means variable.
                Contracts.AssertNonEmpty(srcIndices);
                Contracts.Assert(Utils.Size(srcTypes) == srcIndices.Length);

                SrcSize = srcSize;
                SrcIndices = srcIndices;
                SrcTypes = srcTypes;
            }
        }

        public readonly ColInfo[] Infos;

        protected ManyToOneColumnBindingsBase(ManyToOneColumn[] column, ISchema input, Func<ColumnType[], string> testTypes)
            : base(input, true, GetNamesAndSanitize(column))
        {
            Contracts.AssertNonEmpty(column);
            Contracts.Assert(column.Length == InfoCount);

            // In lieu of actual protections, I have the following silly asserts, so we can have some
            // warning if we decide to rename this argument, and so know to change the below hard-coded
            // standard column name.
            const string standardColumnArgName = "Column";
            Contracts.Assert(nameof(TermTransform.Arguments.Column) == standardColumnArgName);
            Contracts.Assert(nameof(ConcatTransform.Arguments.Column) == standardColumnArgName);

            Infos = new ColInfo[InfoCount];
            for (int i = 0; i < Infos.Length; i++)
            {
                var item = column[i];
                Contracts.AssertNonEmpty(item.Name);
                Contracts.AssertNonEmpty(item.Source);

                var src = item.Source;
                var srcIndices = new int[src.Length];
                var srcTypes = new ColumnType[src.Length];
                int? srcSize = 0;
                for (int j = 0; j < src.Length; j++)
                {
                    Contracts.CheckUserArg(!string.IsNullOrWhiteSpace(src[j]), nameof(ManyToOneColumn.Source));
#pragma warning disable MSML_ContractsNameUsesNameof // Unfortunately, there is no base class for the columns bindings.
                    if (!input.TryGetColumnIndex(src[j], out srcIndices[j]))
                        throw Contracts.ExceptUserArg(standardColumnArgName, "Source column '{0}' not found", src[j]);
#pragma warning restore MSML_ContractsNameUsesNameof
                    srcTypes[j] = input.GetColumnType(srcIndices[j]);
                    var size = srcTypes[j].ValueCount;
                    srcSize = size == 0 ? null : checked(srcSize + size);
                }

                if (testTypes != null)
                {
                    string reason = testTypes(srcTypes);
                    if (reason != null)
                    {
#pragma warning disable MSML_ContractsNameUsesNameof // Unfortunately, there is no base class for the columns bindings.
                        throw Contracts.ExceptUserArg(standardColumnArgName, "Column '{0}' has invalid source types: {1}. Source types: '{2}'.",
                            item.Name, reason, string.Join(", ", srcTypes.Select(type => type.ToString())));
#pragma warning restore MSML_ContractsNameUsesNameof
                    }
                }
                Infos[i] = new ColInfo(srcSize.GetValueOrDefault(), srcIndices, srcTypes);
            }
        }

        /// <summary>
        /// Gather the names from the <see cref="ManyToOneColumn"/> objects. Also, cleanse the column
        /// objects (propagate values that are shared).
        /// </summary>
        private static string[] GetNamesAndSanitize(ManyToOneColumn[] column)
        {
            Contracts.CheckUserArg(Utils.Size(column) > 0, nameof(column));

            var names = new string[column.Length];
            for (int i = 0; i < column.Length; i++)
            {
                var item = column[i];
                if (string.IsNullOrWhiteSpace(item.Name))
                {
                    Contracts.CheckUserArg(Utils.Size(item.Source) > 0, nameof(item.Name), "Must specify name");
                    Contracts.CheckUserArg(item.Source.Length == 1, nameof(item.Name), "New name is required when multiple source columns are specified");
                    item.Name = item.Source[0];
                }
                else if (Utils.Size(item.Source) == 0)
                    item.Source = new string[] { item.Name };
                names[i] = item.Name;
            }

            return names;
        }

        // Read everything into a new Contents object and pass it to the constructor below.
        protected ManyToOneColumnBindingsBase(ModelLoadContext ctx, ISchema input, Func<ColumnType[], string> testTypes)
            : this(new Contents(ctx, input, testTypes))
        {
        }

        private ManyToOneColumnBindingsBase(Contents contents)
            : base(contents.Input, false, contents.Names)
        {
            Contracts.Assert(InfoCount == Utils.Size(contents.Infos));
            Infos = contents.Infos;
        }

        /// <summary>
        /// This class is used to deserialize. We read everything into an instance of this
        /// and pass that to another constructor.
        /// </summary>
        private sealed class Contents
        {
            public ISchema Input;
            public ColInfo[] Infos;
            public string[] Names;

            public Contents(ModelLoadContext ctx, ISchema input, Func<ColumnType[], string> testTypes)
            {
                Contracts.CheckValue(ctx, nameof(ctx));
                Contracts.CheckValue(input, nameof(input));
                Contracts.CheckValueOrNull(testTypes);

                Input = input;

                // *** Binary format ***
                // int: number of added columns
                // for each added column
                //   int: id of output column name
                //   int: number of input column names
                //   int[]: ids of input column names
                int cinfo = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(cinfo > 0);

                Infos = new ColInfo[cinfo];
                Names = new string[cinfo];
                for (int i = 0; i < cinfo; i++)
                {
                    Names[i] = ctx.LoadNonEmptyString();

                    int csrc = ctx.Reader.ReadInt32();
                    Contracts.CheckDecode(csrc > 0);
                    int[] indices = new int[csrc];
                    var srcTypes = new ColumnType[csrc];
                    int? srcSize = 0;
                    for (int j = 0; j < csrc; j++)
                    {
                        string src = ctx.LoadNonEmptyString();
                        if (!input.TryGetColumnIndex(src, out indices[j]))
                            throw Contracts.Except("Source column '{0}' is required but not found", src);
                        srcTypes[j] = input.GetColumnType(indices[j]);
                        var size = srcTypes[j].ValueCount;
                        srcSize = size == 0 ? null : checked(srcSize + size);
                    }

                    if (testTypes != null)
                    {
                        string reason = testTypes(srcTypes);
                        if (reason != null)
                        {
                            throw Contracts.Except("Source columns '{0}' have invalid types: {1}. Source types: '{2}'.",
                                string.Join(", ", indices.Select(k => input.GetColumnName(k))),
                                reason,
                                string.Join(", ", srcTypes.Select(type => type.ToString())));
                        }
                    }

                    Infos[i] = new ColInfo(srcSize.GetValueOrDefault(), indices, srcTypes);
                }
            }
        }

        public virtual void Save(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);

            // *** Binary format ***
            // int: number of added columns
            // for each added column
            //   int: id of output column name
            //   int: number of input column names
            //   int[]: ids of input column names
            ctx.Writer.Write(Infos.Length);
            for (int i = 0; i < Infos.Length; i++)
            {
                var info = Infos[i];
                ctx.SaveNonEmptyString(GetColumnNameCore(i));
                ctx.Writer.Write(info.SrcIndices.Length);
                foreach (int src in info.SrcIndices)
                    ctx.SaveNonEmptyString(Input.GetColumnName(src));
            }
        }

        public Func<int, bool> GetDependencies(Func<int, bool> predicate)
        {
            Contracts.AssertValue(predicate);

            var active = new bool[Input.ColumnCount];
            for (int col = 0; col < ColumnCount; col++)
            {
                if (!predicate(col))
                    continue;

                bool isSrc;
                int index = MapColumnIndex(out isSrc, col);
                if (isSrc)
                    active[index] = true;
                else
                {
                    foreach (var i in Infos[index].SrcIndices)
                        active[i] = true;
                }
            }

            return col => 0 <= col && col < active.Length && active[col];
        }
    }

    /// <summary>
    /// Parsing utilities for converting between transform column argument objects and
    /// command line representations.
    /// </summary>
    public static class ColumnParsingUtils
    {
        /// <summary>
        /// For parsing name and source from a string. This supports "name" and "name:source".
        /// </summary>
        public static bool TryParse(string str, out string name, out string source)
        {
            Contracts.CheckNonWhiteSpace(str, nameof(str));

            int ich = str.IndexOf(':');
            if (ich < 0)
            {
                name = str;
                source = str;
                return true;
            }

            if (0 < ich && ich < str.Length - 1)
            {
                name = str.Substring(0, ich);
                source = str.Substring(ich + 1);
                return !source.Contains(":");
            }

            name = null;
            source = null;
            return false;
        }

        /// <summary>
        /// For parsing name and source from a string. This supports "name" and "name:source" and "name:extra:source".
        /// For the last form, the out extra parameter is set accordingly. For the other forms, extra is set to null.
        /// </summary>
        public static bool TryParse(string str, out string name, out string source, out string extra)
        {
            Contracts.CheckNonWhiteSpace(str, nameof(str));

            extra = null;
            int ich = str.IndexOf(':');
            if (ich < 0)
            {
                name = str;
                source = str;
                return true;
            }
            if (ich == 0 || ich >= str.Length - 1)
            {
                name = null;
                source = null;
                return false;
            }

            name = str.Substring(0, ich);

            int ichMin = ich + 1;
            ich = str.LastIndexOf(':');
            Contracts.Assert(ich >= ichMin - 1);
            if (ich == ichMin - 1)
            {
                source = str.Substring(ichMin);
                return true;
            }
            if (ich == ichMin || ich >= str.Length - 1)
            {
                name = null;
                source = null;
                return false;
            }

            extra = str.Substring(ichMin, ich - ichMin);
            source = str.Substring(ich + 1);

            return true;
        }
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

[assembly: LoadableClass(TextLoader.Summary, typeof(ILegacyDataLoader), typeof(TextLoader), typeof(TextLoader.Options), typeof(SignatureDataLoader),
    "Text Loader", "TextLoader", "Text", DocName = "loader/TextLoader.md")]

[assembly: LoadableClass(TextLoader.Summary, typeof(ILegacyDataLoader), typeof(TextLoader), null, typeof(SignatureLoadDataLoader),
    "Text Loader", TextLoader.LoaderSignature)]

[assembly: LoadableClass(TextLoader.Summary, typeof(TextLoader), null, typeof(SignatureLoadModel),
    "Text Loader", TextLoader.LoaderSignature)]

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Loads a text file into an IDataView. Supports basic mapping from input columns to <see cref="IDataView"/> columns.
    /// </summary>
    public sealed partial class TextLoader : IDataLoader<IMultiStreamSource>
    {
        /// <summary>
        /// Describes how an input column should be mapped to an <see cref="IDataView"/> column.
        /// </summary>
        public sealed class Column
        {
            // Examples of how a column is defined in command line API:
            // Scalar column of <seealso cref="DataKind"/> I4 sourced from 2nd column
            //      col=ColumnName:I4:1
            // Vector column of <seealso cref="DataKind"/> I4 that contains values from columns 1, 3 to 10
            //     col=ColumnName:I4:1,3-10
            // Key range column of KeyType with underlying storage type U4 that contains values from columns 1, 3 to 10, that can go from 1 to 100 (0 reserved for out of range)
            //     col=ColumnName:U4[100]:1,3-10

            /// <summary>
            /// Describes how an input column should be mapped to an <see cref="IDataView"/> column.
            /// </summary>
            public Column() { }

            /// <summary>
            /// Describes how an input column should be mapped to an <see cref="IDataView"/> column.
            /// </summary>
            /// <param name="name">Name of the column.</param>
            /// <param name="dataKind"><see cref="Data.DataKind"/> of the items in the column.</param>
            /// <param name="index">Index of the column.</param>
            public Column(string name, DataKind dataKind, int index)
                : this(name, dataKind.ToInternalDataKind(), new[] { new Range(index) })
            {
            }

            /// <summary>
            /// Describes how an input column should be mapped to an <see cref="IDataView"/> column.
            /// </summary>
            /// <param name="name">Name of the column.</param>
            /// <param name="dataKind"><see cref="Data.DataKind"/> of the items in the column.</param>
            /// <param name="minIndex">The minimum inclusive index of the column.</param>
            /// <param name="maxIndex">The maximum-inclusive index of the column.</param>
            public Column(string name, DataKind dataKind, int minIndex, int maxIndex)
                : this(name, dataKind.ToInternalDataKind(), new[] { new Range(minIndex, maxIndex) })
            {
            }

            /// <summary>
            /// Describes how an input column should be mapped to an <see cref="IDataView"/> column.
            /// </summary>
            /// <param name="name">Name of the column.</param>
            /// <param name="dataKind"><see cref="Data.DataKind"/> of the items in the column.</param>
            /// <param name="source">Source index range(s) of the column.</param>
            /// <param name="keyCount">For a key column, this defines the range of values.</param>
            public Column(string name, DataKind dataKind, Range[] source, KeyCount keyCount = null)
                : this(name, dataKind.ToInternalDataKind(), source, keyCount)
            {
            }

            /// <summary>
            /// Describes how an input column should be mapped to an <see cref="IDataView"/> column.
            /// </summary>
            /// <param name="name">Name of the column.</param>
            /// <param name="kind"><see cref="InternalDataKind"/> of the items in the column.</param>
            /// <param name="source">Source index range(s) of the column.</param>
            /// <param name="keyCount">For a key column, this defines the range of values.</param>
            private Column(string name, InternalDataKind kind, Range[] source, KeyCount keyCount = null)
            {
                Contracts.CheckValue(name, nameof(name));
                Contracts.CheckValue(source, nameof(source));

                Name = name;
                Type = kind;
                Source = source;
                KeyCount = keyCount;
            }

            /// <summary>
            /// Name of the column.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the column")]
            public string Name;

            /// <summary>
            /// <see cref="InternalDataKind"/> of the items in the column. It defaults to float.
            /// Although <see cref="InternalDataKind"/> is internal, <see cref="Type"/>'s information can be publically accessed by <see cref="DataKind"/>.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Type of the items in the column")]
            [BestFriend]
            internal InternalDataKind Type = InternalDataKind.R4;

            /// <summary>
            /// <see cref="Data.DataKind"/> of the items in the column.
            /// </summary>
            /// It's a public interface to access the information in an internal DataKind.
            public DataKind DataKind
            {
                get { return Type.ToDataKind(); }
                set { Type = value.ToInternalDataKind(); }
            }

            /// <summary>
            /// Source index range(s) of the column.
            /// </summary>
            [Argument(ArgumentType.Multiple, HelpText = "Source index range(s) of the column", ShortName = "src")]
            public Range[] Source;

            /// <summary>
            /// For a key column, this defines the range of values.
            /// </summary>
            [Argument(ArgumentType.Multiple, HelpText = "For a key column, this defines the range of values", ShortName = "key")]
            public KeyCount KeyCount;

            internal static Column Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Column();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            private bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                // Allow name:srcs and name:type:srcs
                var rgstr = str.Split(':');
                if (rgstr.Length < 2 || rgstr.Length > 3)
                    return false;

                int istr = 0;
                if (string.IsNullOrWhiteSpace(Name = rgstr[istr++]))
                    return false;
                if (rgstr.Length == 3)
                {
                    InternalDataKind kind;
                    if (!TypeParsingUtils.TryParseDataKind(rgstr[istr++], out kind, out KeyCount))
                        return false;
                    Type = kind == default ? InternalDataKind.R4 : kind;
                }

                return TryParseSource(rgstr[istr++]);
            }

            private bool TryParseSource(string str) => TryParseSourceEx(str, out Source);

            internal static bool TryParseSourceEx(string str, out Range[] ranges)
            {
                ranges = null;
                var strs = str.Split(',');
                if (str.Length == 0)
                    return false;

                ranges = new Range[strs.Length];
                for (int i = 0; i < strs.Length; i++)
                {
                    if ((ranges[i] = Range.Parse(strs[i])) == null)
                        return false;
                }
                return true;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);

                if (string.IsNullOrWhiteSpace(Name))
                    return false;
                if (CmdQuoter.NeedsQuoting(Name))
                    return false;
                if (Utils.Size(Source) == 0)
                    return false;

                int ich = sb.Length;
                sb.Append(Name);
                sb.Append(':');
                if (Type != default || KeyCount != null)
                {
                    if (Type != default)
                        sb.Append(Type.GetString());
                    if (KeyCount != null)
                    {
                        sb.Append('[');
                        if (!KeyCount.TryUnparse(sb))
                        {
                            sb.Length = ich;
                            return false;
                        }
                        sb.Append(']');
                    }
                    sb.Append(':');
                }
                string pre = "";
                foreach (var src in Source)
                {
                    sb.Append(pre);
                    if (!src.TryUnparse(sb))
                    {
                        sb.Length = ich;
                        return false;
                    }
                    pre = ",";
                }
                return true;
            }

            /// <summary>
            ///  Returns <c>true</c> iff the ranges are disjoint, and each range satisfies 0 &lt;= min &lt;= max.
            /// </summary>
            internal bool IsValid()
            {
                if (Utils.Size(Source) == 0)
                    return false;

                var sortedRanges = Source.OrderBy(x => x.Min).ToList();
                var first = sortedRanges[0];
                if (first.Min < 0 || first.Min > first.Max)
                    return false;

                for (int i = 1; i < sortedRanges.Count; i++)
                {
                    var cur = sortedRanges[i];
                    if (cur.Min > cur.Max)
                        return false;

                    var prev = sortedRanges[i - 1];
                    if (prev.Max == null && (prev.AutoEnd || prev.VariableEnd))
                        return false;
                    if (cur.Min <= prev.Max)
                        return false;
                }

                return true;
            }
        }

        /// <summary>
        /// Specifies the range of indices of input columns that should be mapped to an output column.
        /// </summary>
        public sealed class Range
        {
            public Range() { }

            /// <summary>
            /// A range representing a single value. Will result in a scalar column.
            /// </summary>
            /// <param name="index">The index of the field of the text file to read.</param>
            public Range(int index)
            {
                Contracts.CheckParam(index >= 0, nameof(index), "Must be non-negative");
                Min = index;
                Max = index;
            }

            /// <summary>
            /// A range representing a set of values. Will result in a vector column.
            /// </summary>
            /// <param name="min">The minimum inclusive index of the column.</param>
            /// <param name="max">The maximum-inclusive index of the column. If <c>null</c>
            /// indicates that the <see cref="TextLoader"/> should auto-detect the legnth
            /// of the lines, and read untill the end.</param>
            public Range(int min, int? max)
            {
                Contracts.CheckParam(min >= 0, nameof(min), "Must be non-negative");
                Contracts.CheckParam(!(max < min), nameof(max), "If specified, must be greater than or equal to " + nameof(min));

                Min = min;
                Max = max;
                // Note that without the following being set, in the case where there is a single range
                // where Min == Max, the result will not be a vector valued but a scalar column.
                ForceVector = true;
                AutoEnd = max == null;
            }

            /// <summary>
            ///  The minimum index of the column, inclusive.
            /// </summary>
            [Argument(ArgumentType.Required, HelpText = "First index in the range")]
            public int Min;

            /// <summary>
            /// The maximum index of the column, inclusive. If <see langword="null"/>
            /// indicates that the <see cref="TextLoader"/> should auto-detect the legnth
            /// of the lines, and read untill the end.
            /// If max is specified, the fields <see cref="AutoEnd"/> and <see cref="VariableEnd"/> are ignored.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Last index in the range")]
            public int? Max;

            /// <summary>
            /// Whether this range extends to the end of the line, but should be a fixed number of items.
            /// If <see cref="Max"/> is specified, the fields <see cref="AutoEnd"/> and <see cref="VariableEnd"/> are ignored.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "This range extends to the end of the line, but should be a fixed number of items",
                ShortName = "auto")]
            public bool AutoEnd;

            /// <summary>
            /// Whether this range extends to the end of the line, which can vary from line to line.
            /// If <see cref="Max"/> is specified, the fields <see cref="AutoEnd"/> and <see cref="VariableEnd"/> are ignored.
            /// If <see cref="AutoEnd"/> is <see langword="true"/>, then <see cref="VariableEnd"/> is ignored.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "This range extends to the end of the line, which can vary from line to line",
                ShortName = "var")]
            public bool VariableEnd;

            /// <summary>
            /// Whether this range includes only other indices not specified.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "This range includes only other indices not specified", ShortName = "other")]
            public bool AllOther;

            /// <summary>
            /// Force scalar columns to be treated as vectors of length one.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Force scalar columns to be treated as vectors of length one", ShortName = "vector")]
            public bool ForceVector;

            internal static Range Parse(string str)
            {
                Contracts.AssertNonEmpty(str);

                var res = new Range();
                if (res.TryParse(str))
                    return res;
                return null;
            }

            private bool TryParse(string str)
            {
                Contracts.AssertNonEmpty(str);

                int ich = str.IndexOfAny(new char[] { '-', '~' });
                if (ich < 0)
                {
                    // No "-" or "~". Single integer.
                    if (!int.TryParse(str, out Min))
                        return false;
                    Max = Min;
                    return true;
                }

                AllOther = str[ich] == '~';
                ForceVector = true;

                if (ich == 0)
                {
                    if (!AllOther)
                        return false;

                    Min = 0;
                }
                else if (!int.TryParse(str.Substring(0, ich), out Min))
                    return false;

                string rest = str.Substring(ich + 1);
                if (string.IsNullOrEmpty(rest) || rest == "*")
                {
                    AutoEnd = true;
                    return true;
                }
                if (rest == "**")
                {
                    VariableEnd = true;
                    return true;
                }

                int tmp;
                if (!int.TryParse(rest, out tmp))
                    return false;
                Max = tmp;
                return true;
            }

            internal bool TryUnparse(StringBuilder sb)
            {
                Contracts.AssertValue(sb);
                char dash = AllOther ? '~' : '-';
                if (Min < 0)
                    return false;
                sb.Append(Min);
                if (Max != null)
                {
                    if (Max != Min || ForceVector || AllOther)
                        sb.Append(dash).Append(Max);
                }
                else if (AutoEnd)
                    sb.Append(dash).Append("*");
                else if (VariableEnd)
                    sb.Append(dash).Append("**");
                return true;
            }
        }

        /// <summary>
        /// The settings for <see cref="TextLoader"/>
        /// </summary>
        public class Options
        {
            /// <summary>
            /// Whether the input may include quoted values, which can contain separator characters, colons,
            /// and distinguish empty values from missing values. When true, consecutive separators denote a
            /// missing value and an empty value is denoted by \"\". When false, consecutive separators denote an empty value.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce,
                HelpText =
                    "Whether the input may include quoted values, which can contain separator characters, colons," +
                    " and distinguish empty values from missing values. When true, consecutive separators denote a" +
                    " missing value and an empty value is denoted by \"\". When false, consecutive separators" +
                    " denote an empty value.",
                ShortName = "quote")]
            public bool AllowQuoting = Defaults.AllowQuoting;

            /// <summary>
            /// Whether the input may include sparse representations.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the input may include sparse representations", ShortName = "sparse")]
            public bool AllowSparse = Defaults.AllowSparse;

            /// <summary>
            /// Number of source columns in the text data. Default is that sparse rows contain their size information.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Number of source columns in the text data. Default is that sparse rows contain their size information.",
                ShortName = "size")]
            public int? InputSize;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "Source column separator. Options: tab, space, comma, single character", ShortName = "sep")]
            // this is internal as it only serves the command line interface
            internal string Separator = Defaults.Separator.ToString();

            /// <summary>
            /// The characters that should be used as separators column separator.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, Name = nameof(Separator), Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly, HelpText = "Source column separator.", ShortName = "sep")]
            public char[] Separators = new[] { Defaults.Separator };

            /// <summary>
            /// Specifies the input columns that should be mapped to <see cref="IDataView"/> columns.
            /// </summary>
            [Argument(ArgumentType.Multiple, HelpText = "Column groups. Each group is specified as name:type:numeric-ranges, eg, col=Features:R4:1-17,26,35-40",
                Name = "Column", ShortName = "col", SortOrder = 1)]
            public Column[] Columns;

            /// <summary>
            /// Wheter to remove trailing whitespace from lines.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Remove trailing whitespace from lines", ShortName = "trim")]
            public bool TrimWhitespace = Defaults.TrimWhitespace;

            /// <summary>
            /// Whether the data file has a header with feature names.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, ShortName = "header",
                HelpText = "Data file has header with feature names. Header is read only if options 'hs' and 'hf' are not specified.")]
            public bool HasHeader;

            /// <summary>
            /// Whether to use separate parsing threads.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Use separate parsing threads?", ShortName = "threads", Hide = true)]
            public bool UseThreads = true;

            /// <summary>
            /// File containing a header with feature names. If specified, the header defined in the data file is ignored regardless of <see cref="HasHeader"/>.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "File containing a header with feature names. If specified, header defined in the data file (header+) is ignored.",
                ShortName = "hf", IsInputFileName = true)]
            public string HeaderFile;

            /// <summary>
            /// Maximum number of rows to produce.
            /// </summary>
            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of rows to produce", ShortName = "rows", Hide = true)]
            public long? MaxRows;

            /// <summary>
            /// Checks that all column specifications are valid (that is, ranges are disjoint and have min&lt;=max).
            /// </summary>
            internal bool IsValid()
            {
                return Utils.Size(Columns) == 0 || Columns.All(x => x.IsValid());
            }
        }

        internal static class Defaults
        {
            internal const bool AllowQuoting = false;
            internal const bool AllowSparse = false;
            internal const char Separator = '\t';
            internal const bool HasHeader = false;
            internal const bool TrimWhitespace = false;
        }

        /// <summary>
        /// Used as an input column range.
        /// A variable length segment (extending to the end of the input line) is represented by Lim == SrcLim.
        /// </summary>
        internal struct Segment
        {
            public int Min;
            public int Lim;
            public bool ForceVector;

            public bool IsVariable { get { return Lim == SrcLim; } }

            /// <summary>
            /// Be careful with this ctor. lim == SrcLim means that this segment extends to
            /// the end of the input line. If that is not the intent, pass in Min(lim, SrcLim - 1).
            /// </summary>
            public Segment(int min, int lim, bool forceVector)
            {
                Contracts.Assert(0 <= min & min < lim & lim <= SrcLim);
                Min = min;
                Lim = lim;
                ForceVector = forceVector;
            }

            /// <summary>
            /// Defines a segment that extends from min to the end of input.
            /// </summary>
            public Segment(int min)
            {
                Contracts.Assert(0 <= min & min < SrcLim);
                Min = min;
                Lim = SrcLim;
                ForceVector = true;
            }
        }

        /// <summary>
        /// Information for an output column.
        /// </summary>
        internal sealed class ColInfo
        {
            public readonly string Name;
            // REVIEW: Fix this for keys.
            public readonly InternalDataKind Kind;
            public readonly DataViewType ColType;
            public readonly Segment[] Segments;

            // There is at most one variable sized segment, the one at IsegVariable (-1 if none).
            // BaseSize is the sum of the sizes of non-variable segments.
            public readonly int IsegVariable;
            public readonly int SizeBase;

            private ColInfo(string name, DataViewType colType, Segment[] segs, int isegVar, int sizeBase)
            {
                Contracts.AssertNonEmpty(name);
                Contracts.AssertNonEmpty(segs);
                Contracts.Assert(sizeBase > 0 || isegVar >= 0);
                Contracts.Assert(isegVar >= -1);

                Name = name;
                Kind = colType.GetItemType().GetRawKind();
                Contracts.Assert(Kind != 0);
                ColType = colType;
                Segments = segs;
                SizeBase = sizeBase;
                IsegVariable = isegVar;
            }

            public static ColInfo Create(string name, PrimitiveDataViewType itemType, Segment[] segs, bool user)
            {
                Contracts.AssertNonEmpty(name);
                Contracts.AssertValue(itemType);
                Contracts.AssertNonEmpty(segs);

                var order = Utils.GetIdentityPermutation(segs.Length);
                Array.Sort(order, (x, y) => segs[x].Min.CompareTo(segs[y].Min));

                // Check that the segments are disjoint.
                // REVIEW: Should we insist that they are disjoint? Is there any reason to allow overlapping?
                for (int i = 1; i < order.Length; i++)
                {
                    int a = order[i - 1];
                    int b = order[i];
                    Contracts.Assert(segs[a].Min <= segs[b].Min);
                    if (segs[a].Lim > segs[b].Min)
                    {
                        throw user ?
                            Contracts.ExceptUserArg(nameof(Column.Source), "Intervals specified for column '{0}' overlap", name) :
                            Contracts.ExceptDecode("Intervals specified for column '{0}' overlap", name);
                    }
                }

                // Note: since we know that the segments don't overlap, we're guaranteed that
                // the sum of their sizes doesn't overflow.
                int isegVar = -1;
                int size = 0;
                for (int i = 0; i < segs.Length; i++)
                {
                    var seg = segs[i];
                    if (seg.IsVariable)
                    {
                        Contracts.Assert(isegVar == -1);
                        isegVar = i;
                    }
                    else
                        size += seg.Lim - seg.Min;
                }
                Contracts.Assert(size >= segs.Length || size >= segs.Length - 1 && isegVar >= 0);

                DataViewType type = itemType;
                if (isegVar >= 0)
                    type = new VectorType(itemType);
                else if (size > 1 || segs[0].ForceVector)
                    type = new VectorType(itemType, size);

                return new ColInfo(name, type, segs, isegVar, size);
            }
        }

        private sealed class Bindings
        {
            /// <summary>
            /// <see cref="Infos"/>[i] stores the i-th column's name and type. Columns are loaded from the input text file.
            /// </summary>
            public readonly ColInfo[] Infos;
            /// <summary>
            /// <see cref="Infos"/>[i] stores the i-th column's metadata, named <see cref="AnnotationUtils.Kinds.SlotNames"/>
            /// in <see cref="DataViewSchema.Annotations"/>.
            /// </summary>
            private readonly VBuffer<ReadOnlyMemory<char>>[] _slotNames;
            /// <summary>
            /// Empty if <see cref="Options.HasHeader"/> is <see langword="false"/>, no header presents, or upon load
            /// there was no header stored in the model.
            /// </summary>
            private readonly ReadOnlyMemory<char> _header;

            public DataViewSchema OutputSchema { get; }

            public Bindings(TextLoader parent, Column[] cols, IMultiStreamSource headerFile, IMultiStreamSource dataSample)
            {
                Contracts.AssertNonEmpty(cols);
                Contracts.AssertValueOrNull(headerFile);
                Contracts.AssertValueOrNull(dataSample);

                using (var ch = parent._host.Start("Binding"))
                {
                    // Make sure all columns have at least one source range.
                    // Also determine if any columns have a range that extends to the end. If so, then we need
                    // to look at some data to determine the number of source columns.
                    bool needInputSize = false;
                    foreach (var col in cols)
                    {
                        if (Utils.Size(col.Source) == 0)
                            throw ch.ExceptUserArg(nameof(Column.Source), "Must specify some source column indices");
                        if (!needInputSize && col.Source.Any(r => r.AutoEnd && r.Max == null))
                            needInputSize = true;
                    }

                    int inputSize = parent._inputSize;
                    ch.Assert(0 <= inputSize & inputSize < SrcLim);
                    List<ReadOnlyMemory<char>> lines = null;
                    if (headerFile != null)
                        Cursor.GetSomeLines(headerFile, 1, ref lines);
                    if (needInputSize && inputSize == 0)
                        Cursor.GetSomeLines(dataSample, 100, ref lines);
                    else if (headerFile == null && parent.HasHeader)
                        Cursor.GetSomeLines(dataSample, 1, ref lines);

                    if (needInputSize && inputSize == 0)
                    {
                        int min = 0;
                        int max = 0;
                        if (Utils.Size(lines) > 0)
                            Parser.GetInputSize(parent, lines, out min, out max);
                        if (max == 0)
                            throw ch.ExceptUserArg(nameof(Column.Source), "Can't determine the number of source columns without valid data");
                        ch.Assert(min <= max);
                        if (min < max)
                            throw ch.ExceptUserArg(nameof(Column.Source), "The size of input lines is not consistent");
                        // We reserve SrcLim for variable.
                        inputSize = Math.Min(min, SrcLim - 1);
                    }

                    int iinfoOther = -1;
                    PrimitiveDataViewType typeOther = null;
                    Segment[] segsOther = null;
                    int isegOther = -1;

                    Infos = new ColInfo[cols.Length];

                    // This dictionary is used only for detecting duplicated column names specified by user.
                    var nameToInfoIndex = new Dictionary<string, int>(Infos.Length);

                    for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                    {
                        var col = cols[iinfo];

                        ch.CheckNonWhiteSpace(col.Name, nameof(col.Name));
                        string name = col.Name.Trim();
                        if (iinfo == nameToInfoIndex.Count && nameToInfoIndex.ContainsKey(name))
                            ch.Info("Duplicate name(s) specified - later columns will hide earlier ones");

                        PrimitiveDataViewType itemType;
                        InternalDataKind kind;
                        if (col.KeyCount != null)
                        {
                            itemType = TypeParsingUtils.ConstructKeyType(col.Type, col.KeyCount);
                        }
                        else
                        {
                            kind = col.Type == default ? InternalDataKind.R4 : col.Type;
                            ch.CheckUserArg(Enum.IsDefined(typeof(InternalDataKind), kind), nameof(Column.Type), "Bad item type");
                            itemType = ColumnTypeExtensions.PrimitiveTypeFromKind(kind);
                        }

                        // This was checked above.
                        ch.Assert(Utils.Size(col.Source) > 0);

                        var segs = new Segment[col.Source.Length];
                        for (int i = 0; i < segs.Length; i++)
                        {
                            var range = col.Source[i];

                            // Check for remaining range, raise flag.
                            if (range.AllOther)
                            {
                                ch.CheckUserArg(iinfoOther < 0, nameof(Range.AllOther), "At most one all other range can be specified");
                                iinfoOther = iinfo;
                                isegOther = i;
                                typeOther = itemType;
                                segsOther = segs;
                            }

                            // Falling through this block even if range.allOther is true to capture range information.
                            int min = range.Min;
                            ch.CheckUserArg(0 <= min && min < SrcLim - 1, nameof(range.Min));

                            Segment seg;
                            if (range.Max != null)
                            {
                                int max = range.Max.Value;
                                ch.CheckUserArg(min <= max && max < SrcLim - 1, nameof(range.Max));
                                seg = new Segment(min, max + 1, range.ForceVector);
                                ch.Assert(!seg.IsVariable);
                            }
                            else if (range.AutoEnd)
                            {
                                ch.Assert(needInputSize && 0 < inputSize && inputSize < SrcLim);
                                if (min >= inputSize)
                                    throw ch.ExceptUserArg(nameof(range.Min), "Column #{0} not found in the dataset (it only has {1} columns)", min, inputSize);
                                seg = new Segment(min, inputSize, true);
                                ch.Assert(!seg.IsVariable);
                            }
                            else if (range.VariableEnd)
                            {
                                seg = new Segment(min);
                                ch.Assert(seg.IsVariable);
                            }
                            else
                            {
                                seg = new Segment(min, min + 1, range.ForceVector);
                                ch.Assert(!seg.IsVariable);
                            }

                            segs[i] = seg;
                        }

                        // Defer ColInfo generation if the column contains all other indexes.
                        if (iinfoOther != iinfo)
                            Infos[iinfo] = ColInfo.Create(name, itemType, segs, true);

                        nameToInfoIndex[name] = iinfo;
                    }

                    // Note that segsOther[isegOther] is not a real segment to be included.
                    // It only persists segment information such as Min, Max, autoEnd, variableEnd for later processing.
                    // Process all other range.
                    if (iinfoOther >= 0)
                    {
                        ch.Assert(0 <= isegOther && isegOther < segsOther.Length);

                        // segsAll is the segments from all columns.
                        var segsAll = new List<Segment>();
                        for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                        {
                            if (iinfo == iinfoOther)
                                segsAll.AddRange(segsOther.Where((s, i) => i != isegOther));
                            else
                                segsAll.AddRange(Infos[iinfo].Segments);
                        }

                        // segsNew is where we build the segs for the column iinfoOther.
                        var segsNew = new List<Segment>();
                        var segOther = segsOther[isegOther];
                        for (int i = 0; i < segsOther.Length; i++)
                        {
                            if (i != isegOther)
                            {
                                segsNew.Add(segsOther[i]);
                                continue;
                            }

                            // Sort all existing segments by Min, there is no guarantee that segments do not overlap.
                            segsAll.Sort((s1, s2) => s1.Min.CompareTo(s2.Min));

                            int min = segOther.Min;
                            int lim = segOther.Lim;

                            foreach (var seg in segsAll)
                            {
                                // At this step, all indices less than min is contained in some segment, either in
                                // segsAll or segsNew.
                                ch.Assert(min < lim);
                                if (min < seg.Min)
                                    segsNew.Add(new Segment(min, seg.Min, true));
                                if (min < seg.Lim)
                                    min = seg.Lim;
                                if (min >= lim)
                                    break;
                            }

                            if (min < lim)
                                segsNew.Add(new Segment(min, lim, true));
                        }

                        ch.CheckUserArg(segsNew.Count > 0, nameof(Range.AllOther), "No index is selected as all other indexes.");
                        Infos[iinfoOther] = ColInfo.Create(cols[iinfoOther].Name.Trim(), typeOther, segsNew.ToArray(), true);
                    }

                    _slotNames = new VBuffer<ReadOnlyMemory<char>>[Infos.Length];
                    if ((parent.HasHeader || headerFile != null) && Utils.Size(lines) > 0)
                        _header = lines[0];

                    if (!_header.IsEmpty)
                        Parser.ParseSlotNames(parent, _header, Infos, _slotNames);
                }
                OutputSchema = ComputeOutputSchema();
            }

            public Bindings(ModelLoadContext ctx, TextLoader parent)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: number of columns
                // foreach column:
                //   int: id of column name
                //   byte: DataKind
                //   byte: bool of whether this is a key type
                //   for a key type:
                //     ulong: count for key range
                //   int: number of segments
                //   foreach segment:
                //     int: min
                //     int: lim
                //     byte: force vector (verWrittenCur: verIsVectorSupported)
                int cinfo = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(cinfo > 0);
                Infos = new ColInfo[cinfo];

                // This dictionary is used only for detecting duplicated column names specified by user.
                var nameToInfoIndex = new Dictionary<string, int>(Infos.Length);

                for (int iinfo = 0; iinfo < cinfo; iinfo++)
                {
                    string name = ctx.LoadNonEmptyString();

                    PrimitiveDataViewType itemType;
                    var kind = (InternalDataKind)ctx.Reader.ReadByte();
                    Contracts.CheckDecode(Enum.IsDefined(typeof(InternalDataKind), kind));
                    bool isKey = ctx.Reader.ReadBoolByte();
                    if (isKey)
                    {
                        ulong count;
                        Contracts.CheckDecode(KeyType.IsValidDataType(kind.ToType()));

                        // Special treatment for versions that had Min and Contiguous fields in KeyType.
                        if (ctx.Header.ModelVerWritten < VersionNoMinCount)
                        {
                            bool isContig = ctx.Reader.ReadBoolByte();
                            // We no longer support non contiguous values and non zero Min for KeyType.
                            Contracts.CheckDecode(isContig);
                            ulong min = ctx.Reader.ReadUInt64();
                            Contracts.CheckDecode(min == 0);
                            int cnt = ctx.Reader.ReadInt32();
                            Contracts.CheckDecode(cnt >= 0);
                            count = (ulong)cnt;
                            // Since we removed the notion of unknown cardinality (count == 0), we map to the maximum value.
                            if (count == 0)
                                count = kind.ToMaxInt();
                        }
                        else
                        {
                            count = ctx.Reader.ReadUInt64();
                            Contracts.CheckDecode(0 < count);
                        }
                        itemType = new KeyType(kind.ToType(), count);
                    }
                    else
                        itemType = ColumnTypeExtensions.PrimitiveTypeFromKind(kind);

                    int cseg = ctx.Reader.ReadInt32();
                    Contracts.CheckDecode(cseg > 0);
                    var segs = new Segment[cseg];
                    for (int iseg = 0; iseg < cseg; iseg++)
                    {
                        int min = ctx.Reader.ReadInt32();
                        int lim = ctx.Reader.ReadInt32();
                        Contracts.CheckDecode(0 <= min && min < lim && lim <= SrcLim);
                        bool forceVector = false;
                        if (ctx.Header.ModelVerWritten >= VerForceVectorSupported)
                            forceVector = ctx.Reader.ReadBoolByte();
                        segs[iseg] = new Segment(min, lim, forceVector);
                    }

                    // Note that this will throw if the segments are ill-structured, including the case
                    // of multiple variable segments (since those segments will overlap and overlapping
                    // segments are illegal).
                    Infos[iinfo] = ColInfo.Create(name, itemType, segs, false);
                    nameToInfoIndex[name] = iinfo;
                }

                _slotNames = new VBuffer<ReadOnlyMemory<char>>[Infos.Length];

                string result = null;
                ctx.TryLoadTextStream("Header.txt", reader => result = reader.ReadLine());
                if (!string.IsNullOrEmpty(result))
                    Parser.ParseSlotNames(parent, _header = result.AsMemory(), Infos, _slotNames);

                OutputSchema = ComputeOutputSchema();
            }

            internal void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: number of columns
                // foreach column:
                //   int: id of column name
                //   byte: DataKind
                //   byte: bool of whether this is a key type
                //   for a key type:
                //     ulong: count for key range
                //   int: number of segments
                //   foreach segment:
                //     int: min
                //     int: lim
                //     byte: force vector (verWrittenCur: verIsVectorSupported)
                ctx.Writer.Write(Infos.Length);
                for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                {
                    var info = Infos[iinfo];
                    ctx.SaveNonEmptyString(info.Name);
                    var type = info.ColType.GetItemType();
                    InternalDataKind rawKind = type.GetRawKind();
                    Contracts.Assert((InternalDataKind)(byte)rawKind == rawKind);
                    ctx.Writer.Write((byte)rawKind);
                    ctx.Writer.WriteBoolByte(type is KeyType);
                    if (type is KeyType key)
                        ctx.Writer.Write(key.Count);
                    ctx.Writer.Write(info.Segments.Length);
                    foreach (var seg in info.Segments)
                    {
                        ctx.Writer.Write(seg.Min);
                        ctx.Writer.Write(seg.Lim);
                        ctx.Writer.WriteBoolByte(seg.ForceVector);
                    }
                }

                // Save header in an easily human inspectable separate entry.
                if (!_header.IsEmpty)
                    ctx.SaveTextStream("Header.txt", writer => writer.WriteLine(_header.ToString()));
            }

            private DataViewSchema ComputeOutputSchema()
            {
                var schemaBuilder = new DataViewSchema.Builder();

                // Iterate through all loaded columns. The index i indicates the i-th column loaded.
                for (int i = 0; i < Infos.Length; ++i)
                {
                    var info = Infos[i];
                    // Retrieve the only possible metadata of this class.
                    var names = _slotNames[i];
                    if (names.Length > 0)
                    {
                        // Slot names present! Let's add them.
                        var metadataBuilder = new DataViewSchema.Annotations.Builder();
                        metadataBuilder.AddSlotNames(names.Length, (ref VBuffer<ReadOnlyMemory<char>> value) => names.CopyTo(ref value));
                        schemaBuilder.AddColumn(info.Name, info.ColType, metadataBuilder.ToAnnotations());
                    }
                    else
                        // Slot names is empty.
                        schemaBuilder.AddColumn(info.Name, info.ColType);
                }

                return schemaBuilder.ToSchema();
            }
        }

        internal const string Summary = "Loads text data file.";

        internal const string LoaderSignature = "TextLoader";

        private const uint VerForceVectorSupported = 0x0001000A;
        private const uint VersionNoMinCount = 0x0001000C;

        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "TXTLOADR",
                //verWrittenCur: 0x00010001, // Initial
                //verWrittenCur: 0x00010002, // Added support for header
                //verWrittenCur: 0x00010003, // Support for TypeCode
                //verWrittenCur: 0x00010004, // Added _allowSparse
                //verWrittenCur: 0x00010005, // Changed TypeCode to DataKind
                //verWrittenCur: 0x00010006, // Removed weight column support
                //verWrittenCur: 0x00010007, // Added key type support
                //verWrittenCur: 0x00010008, // Added maxRows
                // verWrittenCur: 0x00010009, // Introduced _flags
                //verWrittenCur: 0x0001000A, // Added ForceVector in Range
                //verWrittenCur: 0x0001000B, // Header now retained if used and present
                verWrittenCur: 0x0001000C, // Removed Min and Contiguous from KeyType
                verReadableCur: 0x0001000A,
                verWeCanReadBack: 0x00010009,
                loaderSignature: LoaderSignature,
                loaderAssemblyName: typeof(TextLoader).Assembly.FullName);
        }

        /// <summary>
        /// Option flags. These values are serialized, so changing the values requires
        /// bumping the version number.
        /// </summary>
        [Flags]
        private enum OptionFlags : uint
        {
            TrimWhitespace = 0x01,
            HasHeader = 0x02,
            AllowQuoting = 0x04,
            AllowSparse = 0x08,

            All = TrimWhitespace | HasHeader | AllowQuoting | AllowSparse
        }

        // This is reserved to mean the range extends to the end (the segment is variable).
        private const int SrcLim = int.MaxValue;

        private readonly bool _useThreads;
        private readonly OptionFlags _flags;
        private readonly long _maxRows;
        // Input size is zero for unknown - determined by the data (including sparse rows).
        private readonly int _inputSize;
        private readonly char[] _separators;
        private readonly Bindings _bindings;

        private readonly Parser _parser;

        private bool HasHeader
        {
            get { return (_flags & OptionFlags.HasHeader) != 0; }
        }

        private readonly IHost _host;
        private const string RegistrationName = "TextLoader";

        /// <summary>
        /// Loads a text file into an <see cref="IDataView"/>. Supports basic mapping from input columns to IDataView columns.
        /// </summary>
        /// <param name="env">The environment to use.</param>
        /// <param name="options">Defines the settings of the load operation.</param>
        /// <param name="dataSample">Allows to expose items that can be used for loading.</param>
        internal TextLoader(IHostEnvironment env, Options options = null, IMultiStreamSource dataSample = null)
        {
            options = options ?? new Options();

            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(options, nameof(options));
            _host.CheckValueOrNull(dataSample);

            if (dataSample == null)
                dataSample = new MultiFileSource(null);

            IMultiStreamSource headerFile = null;
            if (!string.IsNullOrWhiteSpace(options.HeaderFile))
                headerFile = new MultiFileSource(options.HeaderFile);

            var cols = options.Columns;
            bool error;
            if (Utils.Size(cols) == 0 && !TryParseSchema(_host, headerFile ?? dataSample, ref options, out cols, out error))
            {
                if (error)
                    throw _host.Except("TextLoader options embedded in the file are invalid");

                // Default to a single Label and the rest Features.
                // REVIEW: Should probably default to the label being a key, but with what range?
                cols = new Column[2];
                cols[0] = Column.Parse("Label:0");
                _host.AssertValue(cols[0]);
                cols[1] = Column.Parse("Features:1-*");
                _host.AssertValue(cols[1]);
            }
            _host.Assert(Utils.Size(cols) > 0);

            _useThreads = options.UseThreads;

            if (options.TrimWhitespace)
                _flags |= OptionFlags.TrimWhitespace;
            if (headerFile == null && options.HasHeader)
                _flags |= OptionFlags.HasHeader;
            if (options.AllowQuoting)
                _flags |= OptionFlags.AllowQuoting;
            if (options.AllowSparse)
                _flags |= OptionFlags.AllowSparse;

            // REVIEW: This should be persisted (if it should be maintained).
            _maxRows = options.MaxRows ?? long.MaxValue;
            _host.CheckUserArg(_maxRows >= 0, nameof(options.MaxRows));

            // Note that _maxDim == 0 means sparsity is illegal.
            _inputSize = options.InputSize ?? 0;
            _host.Check(_inputSize >= 0, "inputSize");
            if (_inputSize >= SrcLim)
                _inputSize = SrcLim - 1;

            _host.CheckNonEmpty(options.Separator, nameof(options.Separator), "Must specify a separator");

            //Default arg.Separator is tab and default options. Separators is also a '\t'.
            //At a time only one default can be different and whichever is different that will
            //be used.
            if (options.Separators.Length > 1 || options.Separators[0] != '\t')
            {
                var separators = new HashSet<char>();
                foreach (char c in options.Separators)
                    separators.Add(NormalizeSeparator(c.ToString()));

                _separators = separators.ToArray();
            }
            else
            {
                string sep = options.Separator.ToLowerInvariant();
                if (sep == ",")
                    _separators = new char[] { ',' };
                else
                {
                    var separators = new HashSet<char>();
                    foreach (string s in sep.Split(','))
                    {
                        if (string.IsNullOrEmpty(s))
                            continue;

                        char c = NormalizeSeparator(s);
                        separators.Add(c);
                    }
                    _separators = separators.ToArray();

                    // Handling ",,,," case, that .Split() returns empty strings.
                    if (_separators.Length == 0)
                        _separators = new char[] { ',' };
                }
            }

            _bindings = new Bindings(this, cols, headerFile, dataSample);
            _parser = new Parser(this);
        }

        private char NormalizeSeparator(string sep)
        {
            switch (sep)
            {
            case "space":
            case " ":
                return ' ';
            case "tab":
            case "\t":
                return '\t';
            case "comma":
            case ",":
                return ',';
            case "colon":
            case ":":
                _host.CheckUserArg((_flags & OptionFlags.AllowSparse) == 0, nameof(Options.Separator),
                    "When the separator is colon, turn off allowSparse");
                return ':';
            case "semicolon":
            case ";":
                return ';';
            case "bar":
            case "|":
                return '|';
            default:
                char ch = sep[0];
                if (sep.Length != 1 || ch < ' ' || '0' <= ch && ch <= '9' || ch == '"')
                    throw _host.ExceptUserArg(nameof(Options.Separator), "Illegal separator: '{0}'", sep);
                return sep[0];
            }
        }

        // This is a private arguments class needed only for parsing options
        // embedded in a data file.
        private sealed class LoaderHolder
        {
#pragma warning disable 649 // never assigned
            [Argument(ArgumentType.Multiple, SignatureType = typeof(SignatureDataLoader))]
            public IComponentFactory<ILegacyDataLoader> Loader;
#pragma warning restore 649 // never assigned
        }

        /// <summary>
        /// See if we can extract valid arguments from the first data file. If so, update options and set cols to the combined set of columns.
        /// If not, set error to true if there was an error condition.
        /// </summary>
        /// <remarks>
        /// Not all arguments are extracted from the data file. There are three arguments that can vary from iteration to iteration and that are set
        /// directly by the user in the options class. These three arguments are:
        /// <see cref="Options.UseThreads"/>,
        /// <see cref="Options.HeaderFile"/>,
        /// <see cref="Options.MaxRows"/>
        /// </remarks>
        private static bool TryParseSchema(IHost host, IMultiStreamSource files,
            ref Options options, out Column[] cols, out bool error)
        {
            host.AssertValue(host);
            host.AssertValue(files);
            host.CheckValue(options, nameof(options));

            cols = null;
            error = false;

            // Verify that the current schema-defining arguments are default.
            // Get a string representation of the settings for all the fields of the Options class besides the following three
            // UseThreads, HeaderFile, MaxRows, which are set by the user directly.
            string tmp = CmdParser.GetSettings(host, options, new Options()
            {
                // It's fine if the user sets the following three arguments, as they are instance specific.
                // Setting the defaults to the user provided values will avoid these in the output of the call CmdParser.GetSettings.
                UseThreads = options.UseThreads,
                HeaderFile = options.HeaderFile,
                MaxRows = options.MaxRows
            });

            // Try to get the schema information from the file.
            string str = Cursor.GetEmbeddedArgs(files);
            if (string.IsNullOrWhiteSpace(str))
                return false;

            // Parse the extracted information.
            using (var ch = host.Start("Parsing options from file"))
            {
                // If tmp is not empty, this means the user specified some additional arguments in the options or command line,
                // such as quote- or sparse-. Warn them about it, since this means that the columns will not be read from the file.
                if (!string.IsNullOrWhiteSpace(tmp))
                {
                    ch.Warning(
                        "Arguments cannot be embedded in the file and in the command line. The embedded arguments will be ignored");
                    return false;
                }

                error = true;
                LoaderHolder h = new LoaderHolder();
                if (!CmdParser.ParseArguments(host, "loader = " + str, h, msg => ch.Error(msg)))
                    goto LDone;

                ch.Assert(h.Loader == null || h.Loader is ICommandLineComponentFactory);
                var loader = h.Loader as ICommandLineComponentFactory;

                if (loader == null || string.IsNullOrWhiteSpace(loader.Name))
                    goto LDone;

                // Make sure the loader binds to us.
                var info = host.ComponentCatalog.GetLoadableClassInfo<SignatureDataLoader>(loader.Name);
                if (info.Type != typeof(ILegacyDataLoader) || info.ArgType != typeof(Options))
                    goto LDone;

                var optionsNew = new Options();
                // Set the fields of optionsNew to the arguments parsed from the file.
                if (!CmdParser.ParseArguments(host, loader.GetSettingsString(), optionsNew, typeof(Options), msg => ch.Error(msg)))
                    goto LDone;

                // Overwrite the three arguments that vary from iteration to iteration with the values specified by the user in the options class.
                optionsNew.UseThreads = options.UseThreads;
                optionsNew.HeaderFile = options.HeaderFile;
                optionsNew.MaxRows = options.MaxRows;

                cols = optionsNew.Columns;
                if (Utils.Size(cols) == 0)
                    goto LDone;

                error = false;
                options = optionsNew;

                LDone:
                return !error;
            }
        }

        /// <summary>
        /// Checks whether the source contains the valid TextLoader.Options depiction.
        /// </summary>
        internal static bool FileContainsValidSchema(IHostEnvironment env, IMultiStreamSource files, out Options options)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(files, nameof(files));
            options = new Options();
            Column[] cols;
            bool error;
            bool found = TryParseSchema(h, files, ref options, out cols, out error);
            return found && !error && options.IsValid();
        }

        private TextLoader(IHost host, ModelLoadContext ctx)
        {
            Contracts.AssertValue(host, "host");
            host.AssertValue(ctx);

            _host = host;

            // REVIEW: Should we serialize this? It really isn't part of the data model.
            _useThreads = true;

            // *** Binary format ***
            // int: sizeof(Float)
            // long: maxRows
            // int: flags
            // int: inputSize: 0 for determined from data
            // int: number of separators
            // char[]: separators
            // bindings
            int cbFloat = ctx.Reader.ReadInt32();
            host.CheckDecode(cbFloat == sizeof(float));
            _maxRows = ctx.Reader.ReadInt64();
            host.CheckDecode(_maxRows > 0);
            _flags = (OptionFlags)ctx.Reader.ReadUInt32();
            host.CheckDecode((_flags & ~OptionFlags.All) == 0);
            _inputSize = ctx.Reader.ReadInt32();
            host.CheckDecode(0 <= _inputSize && _inputSize < SrcLim);

            // Load and validate all separators.
            _separators = ctx.Reader.ReadCharArray();
            host.CheckDecode(Utils.Size(_separators) > 0);
            const string illegalSeparators = "\0\r\n\"0123456789";

            foreach (char sep in _separators)
            {
                if (illegalSeparators.IndexOf(sep) >= 0)
                    throw host.ExceptDecode();
            }

            if (_separators.Contains(':'))
                host.CheckDecode((_flags & OptionFlags.AllowSparse) == 0);

            _bindings = new Bindings(ctx, this);
            _parser = new Parser(this);
        }

        internal static TextLoader Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(RegistrationName);

            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return h.Apply("Loading Model", ch => new TextLoader(h, ctx));
        }

        // These are legacy constructors needed for ComponentCatalog.
        internal static ILegacyDataLoader Create(IHostEnvironment env, ModelLoadContext ctx, IMultiStreamSource files)
            => (ILegacyDataLoader)Create(env, ctx).Load(files);
        internal static ILegacyDataLoader Create(IHostEnvironment env, Options options, IMultiStreamSource files)
            => (ILegacyDataLoader)new TextLoader(env, options, files).Load(files);

        /// <summary>
        /// Convenience method to create a <see cref="TextLoader"/> and use it to load a specified file.
        /// </summary>
        internal static IDataView LoadFile(IHostEnvironment env, Options options, IMultiStreamSource fileSource)
            => new TextLoader(env, options, fileSource).Load(fileSource);

        void ICanSaveModel.Save(ModelSaveContext ctx)
        {
            _host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: sizeof(Float)
            // long: maxRows
            // int: flags
            // int: inputSize: 0 for determined from data
            // int: number of separators
            // char[]: separators
            // bindings
            ctx.Writer.Write(sizeof(float));
            ctx.Writer.Write(_maxRows);
            _host.Assert((_flags & ~OptionFlags.All) == 0);
            ctx.Writer.Write((uint)_flags);
            _host.Assert(0 <= _inputSize && _inputSize < SrcLim);
            ctx.Writer.Write(_inputSize);
            ctx.Writer.WriteCharArray(_separators);

            _bindings.Save(ctx);
        }

        /// <summary>
        /// The output <see cref="DataViewSchema"/> that will be produced by the loader.
        /// </summary>
        public DataViewSchema GetOutputSchema() => _bindings.OutputSchema;

        /// <summary>
        /// Loads data from <paramref name="source"/> into an <see cref="IDataView"/>.
        /// </summary>
        /// <param name="source">The source from which to load data.</param>
        public IDataView Load(IMultiStreamSource source) => new BoundLoader(this, source);

        internal static TextLoader CreateTextLoader<TInput>(IHostEnvironment host,
           bool hasHeader = Defaults.HasHeader,
           char separator = Defaults.Separator,
           bool allowQuoting = Defaults.AllowQuoting,
           bool supportSparse = Defaults.AllowSparse,
           bool trimWhitespace = Defaults.TrimWhitespace,
           IMultiStreamSource dataSample = null)
        {
            var userType = typeof(TInput);

            var fieldInfos = userType.GetFields(BindingFlags.Public | BindingFlags.Instance);

            var propertyInfos =
                userType
                .GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Where(x => x.CanRead && x.CanWrite && x.GetGetMethod() != null && x.GetSetMethod() != null && x.GetIndexParameters().Length == 0);

            var memberInfos = (fieldInfos as IEnumerable<MemberInfo>).Concat(propertyInfos).ToArray();

            var columns = new List<Column>();

            for (int index = 0; index < memberInfos.Length; index++)
            {
                var memberInfo = memberInfos[index];
                var mappingAttr = memberInfo.GetCustomAttribute<LoadColumnAttribute>();

                host.Assert(mappingAttr != null, $"Field or property {memberInfo.Name} is missing the {nameof(LoadColumnAttribute)} attribute");

                var mappingAttrName = memberInfo.GetCustomAttribute<ColumnNameAttribute>();

                var column = new Column();
                column.Name = mappingAttrName?.Name ?? memberInfo.Name;
                column.Source = mappingAttr.Sources.ToArray();
                InternalDataKind dk;
                switch (memberInfo)
                {
                case FieldInfo field:
                    if (!InternalDataKindExtensions.TryGetDataKind(field.FieldType.IsArray ? field.FieldType.GetElementType() : field.FieldType, out dk))
                        throw Contracts.Except($"Field {memberInfo.Name} is of unsupported type.");

                    break;

                case PropertyInfo property:
                    if (!InternalDataKindExtensions.TryGetDataKind(property.PropertyType.IsArray ? property.PropertyType.GetElementType() : property.PropertyType, out dk))
                        throw Contracts.Except($"Property {memberInfo.Name} is of unsupported type.");
                    break;

                default:
                    Contracts.Assert(false);
                    throw Contracts.ExceptNotSupp("Expected a FieldInfo or a PropertyInfo");
                }

                column.Type = dk;

                columns.Add(column);
            }

            Options options = new Options
            {
                HasHeader = hasHeader,
                Separators = new[] { separator },
                AllowQuoting = allowQuoting,
                AllowSparse = supportSparse,
                TrimWhitespace = trimWhitespace,
                Columns = columns.ToArray()
            };

            return new TextLoader(host, options, dataSample: dataSample);
        }

        private sealed class BoundLoader : ILegacyDataLoader
        {
            private readonly TextLoader _loader;
            private readonly IHost _host;
            private readonly IMultiStreamSource _files;

            public BoundLoader(TextLoader loader, IMultiStreamSource files)
            {
                _loader = loader;
                _host = loader._host.Register(nameof(BoundLoader));
                _files = files;
            }

            public long? GetRowCount()
            {
                // We don't know how many rows there are.
                // REVIEW: Should we try to support RowCount?
                return null;
            }

            // REVIEW: Should we try to support shuffling?
            public bool CanShuffle => false;

            public DataViewSchema Schema => _loader._bindings.OutputSchema;

            public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random rand = null)
            {
                _host.CheckValueOrNull(rand);
                var active = Utils.BuildArray(_loader._bindings.OutputSchema.Count, columnsNeeded);
                return Cursor.Create(_loader, _files, active);
            }

            public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random rand = null)
            {
                _host.CheckValueOrNull(rand);
                var active = Utils.BuildArray(_loader._bindings.OutputSchema.Count, columnsNeeded);
                return Cursor.CreateSet(_loader, _files, active, n);
            }

            void ICanSaveModel.Save(ModelSaveContext ctx) => ((ICanSaveModel)_loader).Save(ctx);
        }
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Float = System.Single;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Core.Data;

[assembly: LoadableClass(TextLoader.Summary, typeof(IDataLoader), typeof(TextLoader), typeof(TextLoader.Arguments), typeof(SignatureDataLoader),
    "Text Loader", "TextLoader", "Text", DocName = "loader/TextLoader.md")]

[assembly: LoadableClass(TextLoader.Summary, typeof(IDataLoader), typeof(TextLoader), null, typeof(SignatureLoadDataLoader),
    "Text Loader", TextLoader.LoaderSignature)]

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Loads a text file into an IDataView. Supports basic mapping from input columns to IDataView columns.
    /// Should accept any file that TlcTextInstances accepts.
    /// </summary>
    public sealed partial class TextLoader : IDataReader<IMultiStreamSource>, ICanSaveModel
    {
        /// <example>
        /// Scalar column of <seealso cref="DataKind"/> I4 sourced from 2nd column
        ///      col=ColumnName:I4:1
        ///
        /// Vector column of <seealso cref="DataKind"/> I4 that contains values from columns 1, 3 to 10
        ///     col=ColumnName:I4:1,3-10
        ///
        /// Key range column of KeyType with underlying storage type U4 that contains values from columns 1, 3 to 10, that can go from 1 to 100 (0 reserved for out of range)
        ///     col=ColumnName:U4[1-100]:1,3-10
        /// </example>
        public sealed class Column
        {
            public Column() { }

            public Column(string name, DataKind? type, int index)
               : this(name, type, new[] { new Range(index) }) { }

            public Column(string name, DataKind? type, int minIndex, int maxIndex)
                : this(name, type, new[] { new Range(minIndex, maxIndex) })
            {
            }

            public Column(string name, DataKind? type, Range[] source, KeyRange keyRange = null)
            {
                Contracts.CheckValue(name, nameof(name));
                Contracts.CheckValue(source, nameof(source));
                Contracts.CheckValueOrNull(keyRange);

                Name = name;
                Type = type;
                Source = source;
                KeyRange = keyRange;
            }

            [Argument(ArgumentType.AtMostOnce, HelpText = "Name of the column")]
            public string Name;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Type of the items in the column")]
            public DataKind? Type;

            [Argument(ArgumentType.Multiple, HelpText = "Source index range(s) of the column", ShortName = "src")]
            public Range[] Source;

            [Argument(ArgumentType.Multiple, HelpText = "For a key column, this defines the range of values", ShortName = "key")]
            public KeyRange KeyRange;

            public static Column Parse(string str)
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
                    DataKind kind;
                    if (!TypeParsingUtils.TryParseDataKind(rgstr[istr++], out kind, out KeyRange))
                        return false;
                    Type = kind == default(DataKind) ? default(DataKind?) : kind;
                }

                return TryParseSource(rgstr[istr++]);
            }

            private bool TryParseSource(string str) => TryParseSourceEx(str, out Source);

            public static bool TryParseSourceEx(string str, out Range[] ranges)
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

            public bool TryUnparse(StringBuilder sb)
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
                if (Type != null || KeyRange != null)
                {
                    if (Type != null)
                        sb.Append(Type.Value.GetString());
                    if (KeyRange != null)
                    {
                        sb.Append('[');
                        if (!KeyRange.TryUnparse(sb))
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
            public bool IsValid()
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
            /// of the lines, and read till the end.</param>
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

            [Argument(ArgumentType.Required, HelpText = "First index in the range")]
            public int Min;

            // If max is specified, the fields autoEnd and variableEnd are ignored.
            // Otherwise, if autoEnd is true, then variableEnd is ignored.
            [Argument(ArgumentType.AtMostOnce, HelpText = "Last index in the range")]
            public int? Max;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "This range extends to the end of the line, but should be a fixed number of items",
                ShortName = "auto")]
            public bool AutoEnd;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "This range extends to the end of the line, which can vary from line to line",
                ShortName = "var")]
            public bool VariableEnd;

            [Argument(ArgumentType.AtMostOnce, HelpText = "This range includes only other indices not specified", ShortName = "other")]
            public bool AllOther;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Force scalar columns to be treated as vectors of length one", ShortName = "vector")]
            public bool ForceVector;

            public static Range Parse(string str)
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

            public bool TryUnparse(StringBuilder sb)
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

        public class ArgumentsCore
        {
            [Argument(ArgumentType.AtMostOnce,
                HelpText =
                    "Whether the input may include quoted values, which can contain separator characters, colons," +
                    " and distinguish empty values from missing values. When true, consecutive separators denote a" +
                    " missing value and an empty value is denoted by \"\". When false, consecutive separators" +
                    " denote an empty value.",
                ShortName = "quote")]
            public bool AllowQuoting = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Whether the input may include sparse representations", ShortName = "sparse")]
            public bool AllowSparse = true;

            [Argument(ArgumentType.AtMostOnce,
                HelpText = "Number of source columns in the text data. Default is that sparse rows contain their size information.",
                ShortName = "size")]
            public int? InputSize;

            [Argument(ArgumentType.AtMostOnce, Visibility = ArgumentAttribute.VisibilityType.CmdLineOnly, HelpText = "Source column separator. Options: tab, space, comma, single character", ShortName = "sep")]
            public string Separator = "tab";

            [Argument(ArgumentType.AtMostOnce, Name = nameof(Separator), Visibility = ArgumentAttribute.VisibilityType.EntryPointsOnly, HelpText = "Source column separator.", ShortName = "sep")]
            public char[] SeparatorChars = new[] { '\t' };

            [Argument(ArgumentType.Multiple, HelpText = "Column groups. Each group is specified as name:type:numeric-ranges, eg, col=Features:R4:1-17,26,35-40",
                ShortName = "col", SortOrder = 1)]
            public Column[] Column;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Remove trailing whitespace from lines", ShortName = "trim")]
            public bool TrimWhitespace;

            [Argument(ArgumentType.AtMostOnce, ShortName = "header",
                HelpText = "Data file has header with feature names. Header is read only if options 'hs' and 'hf' are not specified.")]
            public bool HasHeader;

            /// <summary>
            /// Checks that all column specifications are valid (that is, ranges are disjoint and have min&lt;=max).
            /// </summary>
            public bool IsValid()
            {
                return Utils.Size(Column) == 0 || Column.All(x => x.IsValid());
            }
        }

        public sealed class Arguments : ArgumentsCore
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Use separate parsing threads?", ShortName = "threads", Hide = true)]
            public bool UseThreads = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "File containing a header with feature names. If specified, header defined in the data file (header+) is ignored.",
                ShortName = "hf", IsInputFileName = true)]
            public string HeaderFile;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Maximum number of rows to produce", ShortName = "rows", Hide = true)]
            public long? MaxRows;
        }

        /// <summary>
        /// Used as an input column range.
        /// A variable length segment (extending to the end of the input line) is represented by Lim == SrcLim.
        /// </summary>
        private struct Segment
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
        private sealed class ColInfo
        {
            public readonly string Name;
            // REVIEW: Fix this for keys.
            public readonly DataKind Kind;
            public readonly ColumnType ColType;
            public readonly Segment[] Segments;

            // There is at most one variable sized segment, the one at IsegVariable (-1 if none).
            // BaseSize is the sum of the sizes of non-variable segments.
            public readonly int IsegVariable;
            public readonly int SizeBase;

            private ColInfo(string name, ColumnType colType, Segment[] segs, int isegVar, int sizeBase)
            {
                Contracts.AssertNonEmpty(name);
                Contracts.AssertNonEmpty(segs);
                Contracts.Assert(sizeBase > 0 || isegVar >= 0);
                Contracts.Assert(isegVar >= -1);

                Name = name;
                Kind = colType.ItemType.RawKind;
                Contracts.Assert(Kind != 0);
                ColType = colType;
                Segments = segs;
                SizeBase = sizeBase;
                IsegVariable = isegVar;
            }

            public static ColInfo Create(string name, PrimitiveType itemType, Segment[] segs, bool user)
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

                ColumnType type = itemType;
                if (isegVar >= 0)
                    type = new VectorType(itemType);
                else if (size > 1 || segs[0].ForceVector)
                    type = new VectorType(itemType, size);

                return new ColInfo(name, type, segs, isegVar, size);
            }
        }

        private sealed class Bindings : ISchema
        {
            public readonly ColInfo[] Infos;
            public readonly Dictionary<string, int> NameToInfoIndex;
            private readonly VBuffer<ReadOnlyMemory<char>>[] _slotNames;
            // Empty iff either header+ not set in args, or if no header present, or upon load
            // there was no header stored in the model.
            private readonly ReadOnlyMemory<char> _header;

            private readonly MetadataUtils.MetadataGetter<VBuffer<ReadOnlyMemory<char>>> _getSlotNames;

            public Schema AsSchema { get; }

            private Bindings()
            {
                _getSlotNames = GetSlotNames;
            }

            public Bindings(TextLoader parent, Column[] cols, IMultiStreamSource headerFile, IMultiStreamSource dataSample)
                : this()
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
                    PrimitiveType typeOther = null;
                    Segment[] segsOther = null;
                    int isegOther = -1;

                    Infos = new ColInfo[cols.Length];
                    NameToInfoIndex = new Dictionary<string, int>(Infos.Length);
                    for (int iinfo = 0; iinfo < Infos.Length; iinfo++)
                    {
                        var col = cols[iinfo];

                        ch.CheckNonWhiteSpace(col.Name, nameof(col.Name));
                        string name = col.Name.Trim();
                        if (iinfo == NameToInfoIndex.Count && NameToInfoIndex.ContainsKey(name))
                            ch.Info("Duplicate name(s) specified - later columns will hide earlier ones");

                        PrimitiveType itemType;
                        DataKind kind;
                        if (col.KeyRange != null)
                        {
                            itemType = TypeParsingUtils.ConstructKeyType(col.Type, col.KeyRange);
                        }
                        else
                        {
                            kind = col.Type ?? DataKind.Num;
                            ch.CheckUserArg(Enum.IsDefined(typeof(DataKind), kind), nameof(Column.Type), "Bad item type");
                            itemType = PrimitiveType.FromKind(kind);
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

                        NameToInfoIndex[name] = iinfo;
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
                AsSchema = Schema.Create(this);
            }

            public Bindings(ModelLoadContext ctx, TextLoader parent)
                : this()
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: number of columns
                // foreach column:
                //   int: id of column name
                //   byte: DataKind
                //   byte: bool of whether this is a key type
                //   for a key type:
                //     byte: contiguous key range
                //     ulong: min for key range
                //     int: count for key range
                //   int: number of segments
                //   foreach segment:
                //     int: min
                //     int: lim
                //     byte: force vector (verWrittenCur: verIsVectorSupported)
                int cinfo = ctx.Reader.ReadInt32();
                Contracts.CheckDecode(cinfo > 0);
                Infos = new ColInfo[cinfo];
                NameToInfoIndex = new Dictionary<string, int>(Infos.Length);

                for (int iinfo = 0; iinfo < cinfo; iinfo++)
                {
                    string name = ctx.LoadNonEmptyString();

                    PrimitiveType itemType;
                    var kind = (DataKind)ctx.Reader.ReadByte();
                    Contracts.CheckDecode(Enum.IsDefined(typeof(DataKind), kind));
                    bool isKey = ctx.Reader.ReadBoolByte();
                    if (isKey)
                    {
                        Contracts.CheckDecode(KeyType.IsValidDataKind(kind));

                        bool isContig = ctx.Reader.ReadBoolByte();
                        ulong min = ctx.Reader.ReadUInt64();
                        Contracts.CheckDecode(min >= 0);
                        int count = ctx.Reader.ReadInt32();
                        if (count == 0)
                            itemType = new KeyType(kind, min, 0, isContig);
                        else
                        {
                            Contracts.CheckDecode(isContig);
                            Contracts.CheckDecode(2 <= count && (ulong)count <= kind.ToMaxInt());
                            itemType = new KeyType(kind, min, count);
                        }
                    }
                    else
                        itemType = PrimitiveType.FromKind(kind);

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
                    NameToInfoIndex[name] = iinfo;
                }

                _slotNames = new VBuffer<ReadOnlyMemory<char>>[Infos.Length];

                string result = null;
                ctx.TryLoadTextStream("Header.txt", reader => result = reader.ReadLine());
                if (!string.IsNullOrEmpty(result))
                    Parser.ParseSlotNames(parent, _header = result.AsMemory(), Infos, _slotNames);

                AsSchema = Schema.Create(this);
            }

            public void Save(ModelSaveContext ctx)
            {
                Contracts.AssertValue(ctx);

                // *** Binary format ***
                // int: number of columns
                // foreach column:
                //   int: id of column name
                //   byte: DataKind
                //   byte: bool of whether this is a key type
                //   for a key type:
                //     byte: contiguous key range
                //     ulong: min for key range
                //     int: count for key range
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
                    var type = info.ColType.ItemType;
                    Contracts.Assert((DataKind)(byte)type.RawKind == type.RawKind);
                    ctx.Writer.Write((byte)type.RawKind);
                    ctx.Writer.WriteBoolByte(type.IsKey);
                    if (type.IsKey)
                    {
                        var key = type.AsKey;
                        ctx.Writer.WriteBoolByte(key.Contiguous);
                        ctx.Writer.Write(key.Min);
                        ctx.Writer.Write(key.Count);
                    }
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

            public int ColumnCount
            {
                get { return Infos.Length; }
            }

            public bool TryGetColumnIndex(string name, out int col)
            {
                Contracts.CheckValueOrNull(name);
                return NameToInfoIndex.TryGetValue(name, out col);
            }

            public string GetColumnName(int col)
            {
                Contracts.CheckParam(0 <= col && col < Infos.Length, nameof(col));
                return Infos[col].Name;
            }

            public ColumnType GetColumnType(int col)
            {
                Contracts.CheckParam(0 <= col && col < Infos.Length, nameof(col));
                return Infos[col].ColType;
            }

            public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes(int col)
            {
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));

                var names = _slotNames[col];
                if (names.Length > 0)
                {
                    Contracts.Assert(Infos[col].ColType.VectorSize == names.Length);
                    yield return MetadataUtils.GetSlotNamesPair(names.Length);
                }
            }

            public ColumnType GetMetadataTypeOrNull(string kind, int col)
            {
                Contracts.CheckNonEmpty(kind, nameof(kind));
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));

                switch (kind)
                {
                    case MetadataUtils.Kinds.SlotNames:
                        var names = _slotNames[col];
                        if (names.Length == 0)
                            return null;
                        Contracts.Assert(Infos[col].ColType.VectorSize == names.Length);
                        return MetadataUtils.GetNamesType(names.Length);

                    default:
                        return null;
                }
            }

            public void GetMetadata<TValue>(string kind, int col, ref TValue value)
            {
                Contracts.CheckNonEmpty(kind, nameof(kind));
                Contracts.CheckParam(0 <= col && col < ColumnCount, nameof(col));

                switch (kind)
                {
                    case MetadataUtils.Kinds.SlotNames:
                        _getSlotNames.Marshal(col, ref value);
                        return;

                    default:
                        throw MetadataUtils.ExceptGetMetadata();
                }
            }

            private void GetSlotNames(int col, ref VBuffer<ReadOnlyMemory<char>> dst)
            {
                Contracts.Assert(0 <= col && col < ColumnCount);

                var names = _slotNames[col];
                if (names.Length == 0)
                    throw MetadataUtils.ExceptGetMetadata();

                Contracts.Assert(Infos[col].ColType.VectorSize == names.Length);
                names.CopyTo(ref dst);
            }
        }

        internal const string Summary = "Loads text data file.";

        public const string LoaderSignature = "TextLoader";

        private const uint VerForceVectorSupported = 0x0001000A;

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
                verWrittenCur: 0x0001000B, // Header now retained if used and present
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
        private enum Options : uint
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
        private readonly Options _flags;
        private readonly long _maxRows;
        // Input size is zero for unknown - determined by the data (including sparse rows).
        private readonly int _inputSize;
        private readonly char[] _separators;
        private readonly Bindings _bindings;

        private readonly Parser _parser;

        private bool HasHeader
        {
            get { return (_flags & Options.HasHeader) != 0; }
        }

        private readonly IHost _host;
        private const string RegistrationName = "TextLoader";

        public TextLoader(IHostEnvironment env, Column[] columns, Action<Arguments> advancedSettings, IMultiStreamSource dataSample = null)
            : this(env, MakeArgs(columns, advancedSettings), dataSample)
        {
        }

        private static Arguments MakeArgs(Column[] columns, Action<Arguments> advancedSettings)
        {
            var result = new Arguments { Column = columns };
            advancedSettings?.Invoke(result);
            return result;
        }

        public TextLoader(IHostEnvironment env, Arguments args, IMultiStreamSource dataSample = null)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);

            _host.CheckValue(args, nameof(args));
            _host.CheckValueOrNull(dataSample);

            if (dataSample == null)
                dataSample = new MultiFileSource(null);

            IMultiStreamSource headerFile = null;
            if (!string.IsNullOrWhiteSpace(args.HeaderFile))
                headerFile = new MultiFileSource(args.HeaderFile);

            var cols = args.Column;
            bool error;
            if (Utils.Size(cols) == 0 && !TryParseSchema(_host, headerFile ?? dataSample, ref args, out cols, out error))
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

            _useThreads = args.UseThreads;

            if (args.TrimWhitespace)
                _flags |= Options.TrimWhitespace;
            if (headerFile == null && args.HasHeader)
                _flags |= Options.HasHeader;
            if (args.AllowQuoting)
                _flags |= Options.AllowQuoting;
            if (args.AllowSparse)
                _flags |= Options.AllowSparse;

            // REVIEW: This should be persisted (if it should be maintained).
            _maxRows = args.MaxRows ?? long.MaxValue;
            _host.CheckUserArg(_maxRows >= 0, nameof(args.MaxRows));

            // Note that _maxDim == 0 means sparsity is illegal.
            _inputSize = args.InputSize ?? 0;
            _host.Check(_inputSize >= 0, "inputSize");
            if (_inputSize >= SrcLim)
                _inputSize = SrcLim - 1;

            _host.CheckNonEmpty(args.Separator, nameof(args.Separator), "Must specify a separator");

            //Default arg.Separator is tab and default args.SeparatorChars is also a '\t'.
            //At a time only one default can be different and whichever is different that will
            //be used.
            if (args.SeparatorChars.Length > 1 || args.SeparatorChars[0] != '\t')
            {
                var separators = new HashSet<char>();
                foreach (char c in args.SeparatorChars)
                    separators.Add(NormalizeSeparator(c.ToString()));

                _separators = separators.ToArray();
            }
            else
            {
                string sep = args.Separator.ToLowerInvariant();
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
                    _host.CheckUserArg((_flags & Options.AllowSparse) == 0, nameof(Arguments.Separator),
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
                        throw _host.ExceptUserArg(nameof(Arguments.Separator), "Illegal separator: '{0}'", sep);
                    return sep[0];
            }
        }

        // This is a private arguments class needed only for parsing options
        // embedded in a data file.
        private sealed class LoaderHolder
        {
#pragma warning disable 649 // never assigned
            [Argument(ArgumentType.Multiple, SignatureType = typeof(SignatureDataLoader))]
            public IComponentFactory<IDataLoader> Loader;
#pragma warning restore 649 // never assigned
        }

        // See if we can extract valid arguments from the first data file.
        // If so, update args and set cols to the combined set of columns.
        // If not, set error to true if there was an error condition.
        private static bool TryParseSchema(IHost host, IMultiStreamSource files,
            ref Arguments args, out Column[] cols, out bool error)
        {
            host.AssertValue(host);
            host.AssertValue(files);

            cols = null;
            error = false;

            // Verify that the current schema-defining arguments are default.
            // Get settings just for core arguments, not everything.
            string tmp = CmdParser.GetSettings(host, args, new ArgumentsCore());

            // Try to get the schema information from the file.
            string str = Cursor.GetEmbeddedArgs(files);
            if (string.IsNullOrWhiteSpace(str))
                return false;

            // Parse the extracted information.
            using (var ch = host.Start("Parsing options from file"))
            {
                // If tmp is not empty, this means the user specified some additional arguments in the command line,
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
                if (info.Type != typeof(IDataLoader) || info.ArgType != typeof(Arguments))
                    goto LDone;

                var argsNew = new Arguments();
                // Copy the non-core arguments to the new args (we already know that all the core arguments are default).
                var parsed = CmdParser.ParseArguments(host, CmdParser.GetSettings(host, args, new Arguments()), argsNew);
                ch.Assert(parsed);
                // Copy the core arguments to the new args.
                if (!CmdParser.ParseArguments(host, loader.GetSettingsString(), argsNew, typeof(ArgumentsCore), msg => ch.Error(msg)))
                    goto LDone;

                cols = argsNew.Column;
                if (Utils.Size(cols) == 0)
                    goto LDone;

                error = false;
                args = argsNew;

            LDone:
                return !error;
            }
        }

        /// <summary>
        /// Checks whether the source contains the valid TextLoader.Arguments depiction.
        /// </summary>
        public static bool FileContainsValidSchema(IHostEnvironment env, IMultiStreamSource files, out Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            var h = env.Register(RegistrationName);
            h.CheckValue(files, nameof(files));
            args = new Arguments();
            Column[] cols;
            bool error;
            bool found = TryParseSchema(h, files, ref args, out cols, out error);
            return found && !error && args.IsValid();
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
            host.CheckDecode(cbFloat == sizeof(Float));
            _maxRows = ctx.Reader.ReadInt64();
            host.CheckDecode(_maxRows > 0);
            _flags = (Options)ctx.Reader.ReadUInt32();
            host.CheckDecode((_flags & ~Options.All) == 0);
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
                host.CheckDecode((_flags & Options.AllowSparse) == 0);

            _bindings = new Bindings(ctx, this);
            _parser = new Parser(this);
        }

        public static TextLoader Create(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            IHost h = env.Register(RegistrationName);

            h.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel(GetVersionInfo());

            return h.Apply("Loading Model", ch => new TextLoader(h, ctx));
        }

        // These are legacy constructors needed for ComponentCatalog.
        public static IDataLoader Create(IHostEnvironment env, ModelLoadContext ctx, IMultiStreamSource files)
            => (IDataLoader)Create(env, ctx).Read(files);
        public static IDataLoader Create(IHostEnvironment env, Arguments args, IMultiStreamSource files)
            => (IDataLoader)new TextLoader(env, args, files).Read(files);

        /// <summary>
        /// Convenience method to create a <see cref="TextLoader"/> and use it to read a specified file.
        /// </summary>
        public static IDataView ReadFile(IHostEnvironment env, Arguments args, IMultiStreamSource fileSource)
            => new TextLoader(env, args, fileSource).Read(fileSource);

        public void Save(ModelSaveContext ctx)
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
            ctx.Writer.Write(sizeof(Float));
            ctx.Writer.Write(_maxRows);
            _host.Assert((_flags & ~Options.All) == 0);
            ctx.Writer.Write((uint)_flags);
            _host.Assert(0 <= _inputSize && _inputSize < SrcLim);
            ctx.Writer.Write(_inputSize);
            ctx.Writer.WriteCharArray(_separators);

            _bindings.Save(ctx);
        }

        public Schema GetOutputSchema() => _bindings.AsSchema;

        public IDataView Read(IMultiStreamSource source) => new BoundLoader(this, source);

        public IDataView Read(string path) => Read(new MultiFileSource(path));

        private sealed class BoundLoader : IDataLoader
        {
            private readonly TextLoader _reader;
            private readonly IHost _host;
            private readonly IMultiStreamSource _files;

            public BoundLoader(TextLoader reader, IMultiStreamSource files)
            {
                _reader = reader;
                _host = reader._host.Register(nameof(BoundLoader));
                _files = files;
            }

            public long? GetRowCount(bool lazy = true)
            {
                // We don't know how many rows there are.
                // REVIEW: Should we try to support RowCount?
                return null;
            }

            // REVIEW: Should we try to support shuffling?
            public bool CanShuffle => false;

            public Schema Schema => _reader._bindings.AsSchema;

            public IRowCursor GetRowCursor(Func<int, bool> predicate, IRandom rand = null)
            {
                _host.CheckValue(predicate, nameof(predicate));
                _host.CheckValueOrNull(rand);
                var active = Utils.BuildArray(_reader._bindings.ColumnCount, predicate);
                return Cursor.Create(_reader, _files, active);
            }

            public IRowCursor[] GetRowCursorSet(out IRowCursorConsolidator consolidator,
                Func<int, bool> predicate, int n, IRandom rand = null)
            {
                _host.CheckValue(predicate, nameof(predicate));
                _host.CheckValueOrNull(rand);
                var active = Utils.BuildArray(_reader._bindings.ColumnCount, predicate);
                return Cursor.CreateSet(out consolidator, _reader, _files, active, n);
            }

            public void Save(ModelSaveContext ctx) => _reader.Save(ctx);
        }
    }
}
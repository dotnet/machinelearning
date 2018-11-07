// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Internal.Internallearn;

[assembly: LoadableClass(TextSaver.Summary, typeof(TextSaver), typeof(TextSaver.Arguments), typeof(SignatureDataSaver),
    "Text Saver", "TextSaver", "Text", DocName = "saver/TextSaver.md")]

namespace Microsoft.ML.Runtime.Data.IO
{
    public sealed class TextSaver : IDataSaver
    {
        // REVIEW: consider saving a command line in a separate file.
        public sealed class Arguments
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Separator", ShortName = "sep")]
            public string Separator = "tab";

            [Argument(ArgumentType.AtMostOnce, HelpText = "Force dense format", ShortName = "dense")]
            public bool Dense;

            // REVIEW: This and the corresponding BinarySaver option should be removed,
            // with the silence being handled, somehow, at the environment level. (Task 6158846.)
            [Argument(ArgumentType.LastOccurenceWins, HelpText = "Suppress any info output (not warnings or errors)", Hide = true)]
            public bool Silent;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output the comment containing the loader settings", ShortName = "schema")]
            public bool OutputSchema = true;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Output the header", ShortName = "header")]
            public bool OutputHeader = true;
        }

        internal const string Summary = "Writes data into a text file.";

        private abstract class ValueWriter
        {
            public readonly int Source;

            public static ValueWriter Create(IRowCursor cursor, int col, char sep)
            {
                Contracts.AssertValue(cursor);

                ColumnType type = cursor.Schema.GetColumnType(col);
                Type writePipeType;
                if (type.IsVector)
                    writePipeType = typeof(VecValueWriter<>).MakeGenericType(type.ItemType.RawType);
                else
                    writePipeType = typeof(ValueWriter<>).MakeGenericType(type.RawType);

                return (ValueWriter)Activator.CreateInstance(writePipeType, cursor, type, col, sep);
            }

            public abstract string Default { get; }

            public ValueWriter(int source)
            {
                Source = source;
            }

            /// <summary>
            /// Write the data to the given stream. This requires that FetchData was previously called.
            /// </summary>
            public abstract void WriteData(Action<StringBuilder, int> appendItem, out int length);

            public abstract void WriteHeader(Action<StringBuilder, int> appendItem, out int length);
        }

        private abstract class ValueWriterBase<T> : ValueWriter
        {
            protected readonly ValueMapper<T, StringBuilder> Conv;
            protected readonly char Sep;
            protected StringBuilder Sb;

            public override string Default { get; }

            protected ValueWriterBase(PrimitiveType type, int source, char sep)
                : base(source)
            {
                Contracts.Assert(type.IsStandardScalar || type.IsKey);
                Contracts.Assert(type.RawType == typeof(T));

                Sep = sep;
                if (type.IsText)
                {
                    // For text we need to deal with escaping.
                    ValueMapper<ReadOnlyMemory<char>, StringBuilder> c = MapText;
                    Conv = (ValueMapper<T, StringBuilder>)(Delegate)c;
                }
                else if (type is TimeSpanType)
                {
                    ValueMapper<TimeSpan, StringBuilder> c = MapTimeSpan;
                    Conv = (ValueMapper<T, StringBuilder>)(Delegate)c;
                }
                else if (type is DateTimeType)
                {
                    ValueMapper<DateTime, StringBuilder> c = MapDateTime;
                    Conv = (ValueMapper<T, StringBuilder>)(Delegate)c;
                }
                else if (type is DateTimeOffsetType)
                {
                    ValueMapper<DateTimeOffset, StringBuilder> c = MapDateTimeZone;
                    Conv = (ValueMapper<T, StringBuilder>)(Delegate)c;
                }
                else
                    Conv = Conversions.Instance.GetStringConversion<T>(type);

                var d = default(T);
                Conv(in d, ref Sb);
                Default = Sb.ToString();
            }

            protected void MapText(in ReadOnlyMemory<char> src, ref StringBuilder sb)
            {
                TextSaverUtils.MapText(src.Span, ref sb, Sep);
            }

            protected void MapTimeSpan(in TimeSpan src, ref StringBuilder sb)
            {
                TextSaverUtils.MapTimeSpan(in src, ref sb);
            }

            protected void MapDateTime(in DateTime src, ref StringBuilder sb)
            {
                TextSaverUtils.MapDateTime(in src, ref sb);
            }

            protected void MapDateTimeZone(in DateTimeOffset src, ref StringBuilder sb)
            {
                TextSaverUtils.MapDateTimeZone(in src, ref sb);
            }
        }

        private sealed class VecValueWriter<T> : ValueWriterBase<T>
        {
            private readonly ValueGetter<VBuffer<T>> _getSrc;
            private VBuffer<T> _src;
            private readonly VBuffer<ReadOnlyMemory<char>> _slotNames;
            private readonly int _slotCount;

            public VecValueWriter(IRowCursor cursor, VectorType type, int source, char sep)
                : base(type.ItemType, source, sep)
            {
                _getSrc = cursor.GetGetter<VBuffer<T>>(source);
                ColumnType typeNames;
                if (type.IsKnownSizeVector &&
                    (typeNames = cursor.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, source)) != null &&
                    typeNames.VectorSize == type.VectorSize && typeNames.ItemType.IsText)
                {
                    cursor.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, source, ref _slotNames);
                    Contracts.Check(_slotNames.Length == typeNames.VectorSize, "Unexpected slot names length");
                }
                _slotCount = type.VectorSize;
            }

            public override void WriteData(Action<StringBuilder, int> appendItem, out int length)
            {
                _getSrc(ref _src);
                if (_src.IsDense)
                {
                    for (int i = 0; i < _src.Length; i++)
                    {
                        Conv(in _src.Values[i], ref Sb);
                        appendItem(Sb, i);
                    }
                }
                else
                {
                    for (int i = 0; i < _src.Count; i++)
                    {
                        Conv(in _src.Values[i], ref Sb);
                        appendItem(Sb, _src.Indices[i]);
                    }
                }
                length = _src.Length;
            }

            public override void WriteHeader(Action<StringBuilder, int> appendItem, out int length)
            {
                length = _slotCount;
                if (_slotNames.Count == 0)
                    return;
                for (int i = 0; i < _slotNames.Count; i++)
                {
                    var name = _slotNames.Values[i];
                    if (name.IsEmpty)
                        continue;
                    MapText(in name, ref Sb);
                    int index = _slotNames.IsDense ? i : _slotNames.Indices[i];
                    appendItem(Sb, index);
                }
            }
        }

        private sealed class ValueWriter<T> : ValueWriterBase<T>
        {
            private readonly ValueGetter<T> _getSrc;
            private T _src;
            private string _columnName;

            public ValueWriter(IRowCursor cursor, PrimitiveType type, int source, char sep)
                : base(type, source, sep)
            {
                _getSrc = cursor.GetGetter<T>(source);
                _columnName = cursor.Schema.GetColumnName(source);
            }

            public override void WriteData(Action<StringBuilder, int> appendItem, out int length)
            {
                _getSrc(ref _src);
                Conv(in _src, ref Sb);
                appendItem(Sb, 0);
                length = 1;
            }

            public override void WriteHeader(Action<StringBuilder, int> appendItem, out int length)
            {
                var span = _columnName.AsMemory();
                MapText(in span, ref Sb);
                appendItem(Sb, 0);
                length = 1;
            }
        }

        private readonly bool _forceDense;
        private readonly bool _outputSchema;
        private readonly bool _outputHeader;
        private readonly char _sepChar;
        private readonly string _sepStr;
        private readonly bool _silent;

        private readonly IHost _host;

        // Used to calculate the efficiency of sparse format.
        private const Double _sparseWeight = 2.5;

        public TextSaver(IHostEnvironment env, Arguments args)
        {
            Contracts.AssertValue(args);
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register("TextSaver");
            _forceDense = args.Dense;
            _outputSchema = args.OutputSchema;
            _outputHeader = args.OutputHeader;

            _sepChar = SepStrToChar(args.Separator);
            _sepStr = _sepChar.ToString();
            _silent = args.Silent;
        }

        private static char SepStrToChar(string sep)
        {
            Contracts.CheckUserArg(!string.IsNullOrEmpty(sep), nameof(Arguments.Separator), "Must specify a separator");
            sep = sep.ToLowerInvariant();
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
            case "semicolon":
            case ";":
                return ';';
            case "bar":
            case "|":
                return '|';
            default:
                throw Contracts.ExceptUserArg(nameof(Arguments.Separator), "Invalid separator - must be: space, tab, comma, semicolon, or bar");
            }
        }

        /// <summary>
        /// Returns the string representation of a separator: helpful if it's whitespace or a punctuation mark.
        /// </summary>
        public static string SeparatorCharToString(char separator)
        {
            switch (separator)
            {
            case ' ':
                return "space";
            case '\t':
                return "tab";
            case ',':
                return "comma";
            case ';':
                return "semicolon";
            case '|':
                return "bar";

            default:
                return separator.ToString();
            }
        }

        public bool IsColumnSavable(ColumnType type)
        {
            var item = type.ItemType;
            return item.IsStandardScalar || item.IsKey;
        }

        public void SaveData(Stream stream, IDataView data, params int[] cols)
        {
            string argsLoader;
            SaveData(out argsLoader, stream, data, cols);
        }

        public void SaveData(out string argsLoader, Stream stream, IDataView data, params int[] cols)
        {
            _host.CheckValue(stream, nameof(stream));
            _host.CheckValue(data, nameof(data));
            _host.CheckNonEmpty(cols, nameof(cols));

            using (var ch = _host.Start("Saving"))
            {
                long count;
                int min;
                int max;
                using (var writer = Utils.OpenWriter(stream))
                    WriteDataCore(ch, writer, data, out argsLoader, out count, out min, out max, cols);

                if (!_silent)
                    ShowCount(ch, count, min, max);
            }
        }

        public void WriteData(IDataView data, bool showCount, params int[] cols)
        {
            _host.CheckValue(data, nameof(data));
            _host.CheckNonEmpty(cols, nameof(cols));

            string argsLoader;
            using (var ch = _host.Start("Writing"))
            {
                long count;
                int min;
                int max;
                using (var writer = new StringWriter())
                {
                    WriteDataCore(ch, writer, data, out argsLoader, out count, out min, out max, cols);
                    ch.Info(MessageSensitivity.UserData | MessageSensitivity.Schema, writer.ToString());
                }

                if (showCount)
                    ShowCount(ch, count, min, max);
            }
        }

        private void ShowCount(IChannel ch, long count, int min, int max)
        {
            if (count == 0)
                ch.Warning(MessageSensitivity.None, "Wrote zero rows of data!");
            else if (min == max)
                ch.Info(MessageSensitivity.None, "Wrote {0} rows of length {1}", count, min);
            else
                ch.Info(MessageSensitivity.None, "Wrote {0} rows of lengths between {1} and {2}", count, min, max);
        }

        private void WriteDataCore(IChannel ch, TextWriter writer, IDataView data,
            out string argsLoader, out long count, out int min, out int max, params int[] cols)
        {
            _host.AssertValue(ch);
            ch.AssertValue(writer);
            ch.AssertValue(data);
            ch.AssertNonEmpty(cols);

            // Determine the active columns and whether there is header information.
            bool[] active = new bool[data.Schema.ColumnCount];
            for (int i = 0; i < cols.Length; i++)
            {
                ch.Check(0 <= cols[i] && cols[i] < active.Length);
                ch.Check(data.Schema.GetColumnType(cols[i]).ItemType.RawKind != 0);
                active[cols[i]] = true;
            }

            bool hasHeader = false;
            if (_outputHeader)
            {
                for (int i = 0; i < cols.Length; i++)
                {
                    if (hasHeader)
                        continue;
                    var type = data.Schema.GetColumnType(cols[i]);
                    if (!type.IsVector)
                    {
                        hasHeader = true;
                        continue;
                    }
                    if (!type.IsKnownSizeVector)
                        continue;
                    var typeNames = data.Schema.GetMetadataTypeOrNull(MetadataUtils.Kinds.SlotNames, cols[i]);
                    if (typeNames != null && typeNames.VectorSize == type.VectorSize && typeNames.ItemType.IsText)
                        hasHeader = true;
                }
            }

            using (var cursor = data.GetRowCursor(i => active[i]))
            {
                var pipes = new ValueWriter[cols.Length];
                for (int i = 0; i < cols.Length; i++)
                    pipes[i] = ValueWriter.Create(cursor, cols[i], _sepChar);

                // REVIEW: This should be outside the cursor creation.
                string header = CreateLoaderArguments(data.Schema, pipes, hasHeader, ch);
                argsLoader = header;
                if (_outputSchema)
                    WriteSchemaAsComment(writer, header);

                double rowCount = data.GetRowCount(true) ?? double.NaN;
                using (var pch = !_silent ? _host.StartProgressChannel("TextSaver: saving data") : null)
                {
                    long stateCount = 0;
                    var state = new State(this, writer, pipes, hasHeader);
                    if (pch != null)
                        pch.SetHeader(new ProgressHeader(new[] { "rows" }), e => e.SetProgress(0, stateCount, rowCount));
                    state.Run(cursor, ref stateCount, out min, out max);
                    count = stateCount;
                    if (pch != null)
                        pch.Checkpoint(stateCount);

                }
            }
        }

        private void WriteSchemaAsComment(TextWriter writer, string str)
        {
            writer.WriteLine("#@ TextLoader{");
            foreach (string line in CmdIndenter.GetIndentedCommandLine(str).Split(new[] { "\n", "\r" }, StringSplitOptions.RemoveEmptyEntries))
                writer.WriteLine("#@   " + line);
            writer.WriteLine("#@ }");
        }

        private string CreateLoaderArguments(ISchema schema, ValueWriter[] pipes, bool hasHeader, IChannel ch)
        {
            StringBuilder sb = new StringBuilder();
            if (hasHeader)
                sb.Append("header+ ");
            sb.AppendFormat("sep={0}", SeparatorCharToString(_sepChar));

            // This variable indicates the start index of each column.
            // If null, it means the index cannot be determined.
            int? index = 0;
            for (int i = 0; i < pipes.Length; i++)
            {
                int src = pipes[i].Source;
                string name = schema.GetColumnName(src);
                var type = schema.GetColumnType(src);

                var column = GetColumn(name, type, index);
                sb.Append(" col=");
                if (!column.TryUnparse(sb))
                {
                    var settings = CmdParser.GetSettings(_host, column, new TextLoader.Column());
                    CmdQuoter.QuoteValue(settings, sb, true);
                }
                if (type.IsVector && !type.IsKnownSizeVector && i != pipes.Length - 1)
                {
                    ch.Warning("Column '{0}' is variable length, so it must be the last, or the file will be unreadable. Consider switching to binary format or use xf=Choose to make '{0}' the last column.", name);
                    index = null;
                }

                index += type.ValueCount;
            }

            return sb.ToString();
        }

        private TextLoader.Column GetColumn(string name, ColumnType type, int? start)
        {
            DataKind? kind;
            KeyRange keyRange = null;
            if (type.ItemType.IsKey)
            {
                var key = type.ItemType.AsKey;
                if (!key.Contiguous)
                    keyRange = new KeyRange(key.Min, contiguous: false);
                else if (key.Count == 0)
                    keyRange = new KeyRange(key.Min);
                else
                {
                    Contracts.Assert(key.Count >= 1);
                    keyRange = new KeyRange(key.Min, key.Min + (ulong)(key.Count - 1));
                }
                kind = key.RawKind;
            }
            else
                kind = type.ItemType.RawKind;

            TextLoader.Range[] source = null;

            TextLoader.Range range = null;
            int minValue = start ?? -1;
            if (type.IsKnownSizeVector)
                range = new TextLoader.Range { Min = minValue, Max = minValue + type.ValueCount - 1, ForceVector = true };
            else if (type.IsVector)
                range = new TextLoader.Range { Min = minValue, VariableEnd = true };
            else
                range = new TextLoader.Range { Min = minValue };
            source = new TextLoader.Range[1] { range };
            return new TextLoader.Column() { Name = name, KeyRange = keyRange, Source = source, Type = kind };
        }

        private sealed class State
        {
            private readonly bool _dense;
            private readonly char _sepChar;
            private readonly string _sepStr;
            private readonly TextWriter _writer;
            private readonly ValueWriter[] _pipes;
            private readonly bool _hasHeader;
            private readonly IHost _host;

            // Current column and it's starting destination index.
            private int _col;
            private int _dstBase;
            private int _dstPrev;

            // Map from column to starting destination index and slot.
            private int[] _mpcoldst;
            private int[] _mpcolslot;

            // "slot" is an index into _mpslotdst and _mpslotichLim. _mpslotdst is the sequence of
            // destination indices. _mpslotichLim is the sequence of upper bounds on the characters
            // for the slots.
            private int _slotLim;
            private int[] _mpslotdst;
            private int[] _mpslotichLim;

            // The character buffer.
            private int _cch;
            private char[] _rgch;

            public State(TextSaver parent, TextWriter writer, ValueWriter[] pipes, bool hasHeader)
            {
                Contracts.AssertValue(parent);
                Contracts.AssertValue(parent._host);
                _host = parent._host;
                _host.AssertValue(writer);
                _host.AssertValue(pipes);

                _dense = parent._forceDense;
                _sepChar = parent._sepChar;
                _sepStr = parent._sepStr;

                _writer = writer;
                _pipes = pipes;
                _hasHeader = hasHeader && parent._outputHeader;

                _mpcoldst = new int[_pipes.Length + 1];
                _mpcolslot = new int[_pipes.Length + 1];

                _rgch = new char[1024];
                _mpslotdst = new int[128];
                _mpslotichLim = new int[128];
            }

            public void Run(IRowCursor cursor, ref long count, out int minLen, out int maxLen)
            {
                minLen = int.MaxValue;
                maxLen = 0;

                Action<StringBuilder, int> append = (sb, index) => AppendItem(sb, index, _pipes[_col].Default);
                Action<StringBuilder, int> appendHeader = (sb, index) => AppendItem(sb, index, "");

                if (_hasHeader)
                {
                    StartLine();
                    while (_col < _pipes.Length)
                    {
                        int len;
                        _pipes[_col].WriteHeader(appendHeader, out len);
                        Contracts.Assert(len >= 0);
                        EndColumn(len);
                    }
                    EndLine("\"\"");
                    _writer.WriteLine();
                }

                while (cursor.MoveNext())
                {
                    // Start a new line. This also starts the first column.
                    StartLine();

                    while (_col < _pipes.Length)
                    {
                        int len;
                        _pipes[_col].WriteData(append, out len);
                        Contracts.Assert(len >= 0);
                        EndColumn(len);
                    }

                    if (minLen > _dstBase)
                        minLen = _dstBase;
                    if (maxLen < _dstBase)
                        maxLen = _dstBase;

                    EndLine();
                    _writer.WriteLine();
                    count++;
                }
            }

            private void StartLine()
            {
                _cch = 0;
                _slotLim = 0;
                _col = 0;
                _dstBase = 0;
                _dstPrev = -1;
                Contracts.Assert(_mpcoldst[_col] == 0);
                Contracts.Assert(_mpcolslot[_col] == 0);
            }

            private bool Matches(StringBuilder sb, string def)
            {
                if (sb.Length != def.Length)
                    return false;
                for (int ich = 0; ich < def.Length; ich++)
                {
                    if (sb[ich] != def[ich])
                        return false;
                }
                return true;
            }

            private void AppendItem(StringBuilder sb, int index, string def)
            {
                Contracts.Assert(0 <= _col && _col < _pipes.Length);
                Contracts.AssertValue(sb);

                Contracts.Check(index >= 0);
                int dst = checked(_dstBase + index);
                Contracts.Check(dst > _dstPrev);
                _dstPrev = dst;

                // Suppress default values.
                if (Matches(sb, def))
                    return;

                // Make sure the buffers are big enough.
                int cch = checked(_cch + sb.Length);
                while (cch > _rgch.Length)
                    Array.Resize(ref _rgch, checked(_rgch.Length * 2));
                if (_mpslotdst.Length <= _slotLim)
                    Array.Resize(ref _mpslotdst, checked(_mpslotdst.Length * 2));
                if (_mpslotichLim.Length <= _slotLim)
                    Array.Resize(ref _mpslotichLim, checked(_mpslotichLim.Length * 2));

                // Copy the characters and update the mappings.
                sb.CopyTo(0, _rgch, _cch, sb.Length);
                _cch = cch;
                _mpslotichLim[_slotLim] = _cch;
                _mpslotdst[_slotLim] = dst;
                _slotLim++;
            }

            private void EndColumn(int length)
            {
                Contracts.Assert(_col < _pipes.Length);
                int dst = checked(_dstBase + length);
                Contracts.Check(_dstPrev < dst);
                ++_col;
                _mpcoldst[_col] = dst;
                _mpcolslot[_col] = _slotLim;
                _dstBase = dst;
            }

            private void EndLine(string defaultStr = null)
            {
                Contracts.Assert(_col == _pipes.Length);

                if (_dense)
                {
                    WriteDenseTo(_dstBase, defaultStr);
                    return;
                }

                // Find the sparse split point.
                // REVIEW: Should we allow splitting at any slot or only at column boundaries?
                // This currently does the latter.
                int colBest = 0;
                Double bestScore = _sparseWeight * _slotLim;
                for (int col = 1; col <= _pipes.Length; col++)
                {
                    int cd = _mpcoldst[col];
                    int cs = _slotLim - _mpcolslot[col];

                    Double score = cd + _sparseWeight * cs;
                    if (bestScore > score)
                    {
                        bestScore = score;
                        colBest = col;
                    }
                }

                // If the length of the sparse section is small compared to the dense count,
                // don't bother with sparse.
                int lenDense = _mpcoldst[colBest];
                int lenSparse = _dstBase - lenDense;
                if (lenSparse < 5 || lenSparse < lenDense / 5)
                    colBest = _pipes.Length;

                string sep = "";
                if (colBest > 0)
                {
                    WriteDenseTo(_mpcoldst[colBest], defaultStr);
                    sep = _sepStr;
                }

                if (colBest >= _pipes.Length)
                    return;

                // Write the rest sparsely.
                _writer.Write(sep);
                sep = _sepStr;
                _writer.Write(lenSparse);

                int slot = _mpcolslot[colBest];
                if (slot == _slotLim)
                {
                    // Need to write at least one sparse specification.
                    _writer.Write(sep);
                    _writer.Write("0:");
                    _writer.Write(defaultStr ?? _pipes[colBest].Default);
                    return;
                }

                int ichMin = slot > 0 ? _mpslotichLim[slot - 1] : 0;
                for (; slot < _slotLim; slot++)
                {
                    _writer.Write(sep);
                    _writer.Write(_mpslotdst[slot] - lenDense);
                    _writer.Write(':');
                    int ichLim = _mpslotichLim[slot];
                    _writer.Write(_rgch, ichMin, ichLim - ichMin);
                    ichMin = ichLim;
                }
            }

            private void WriteDenseTo(int dstLim, string defaultStr = null)
            {
                Contracts.Assert(_col == _pipes.Length);
                Contracts.Assert(dstLim <= _dstBase);

                string sep = "";
                int col = 0;
                string def = defaultStr ?? _pipes[0].Default;
                int slot = 0;
                int ichMin = 0;
                for (int dst = 0; dst < dstLim; dst++)
                {
                    Contracts.Assert(slot == _slotLim || slot < _slotLim && dst <= _mpslotdst[slot]);

                    _writer.Write(sep);
                    sep = _sepStr;
                    while (dst >= _mpcoldst[col + 1])
                    {
                        // Move to the next column
                        col++;
                        Contracts.Assert(col < _pipes.Length);
                        def = defaultStr ?? _pipes[col].Default;
                    }

                    if (slot == _slotLim || dst < _mpslotdst[slot])
                        _writer.Write(def);
                    else
                    {
                        int ichLim = _mpslotichLim[slot];
                        _writer.Write(_rgch, ichMin, ichLim - ichMin);
                        ichMin = ichLim;
                        slot++;
                    }
                }
            }
        }
    }

    internal static class TextSaverUtils
    {
        /// <summary>
        /// Converts a ReadOnlySpan to a StringBuilder using TextSaver escaping and string quoting rules.
        /// </summary>
        internal static void MapText(ReadOnlySpan<char> span, ref StringBuilder sb, char sep)
        {
            if (sb == null)
                sb = new StringBuilder();
            else
                sb.Clear();

            if (span.IsEmpty)
                sb.Append("\"\"");
            else
            {
                int ichMin = 0;
                int ichLim = span.Length;
                int ichCur = ichMin;
                int ichRun = ichCur;
                bool quoted = false;

                // Strings that start with space need to be quoted.
                Contracts.Assert(ichCur < ichLim);
                if (span[ichCur] == ' ')
                {
                    quoted = true;
                    sb.Append('"');
                }

                for (; ichCur < ichLim; ichCur++)
                {
                    char ch = span[ichCur];
                    if (ch != '"' && ch != sep && ch != ':')
                        continue;
                    if (!quoted)
                    {
                        Contracts.Assert(ichRun == ichMin);
                        sb.Append('"');
                        quoted = true;
                    }
                    if (ch == '"')
                    {
                        if (ichRun < ichCur)
                            sb.AppendSpan(span.Slice(ichRun, ichCur - ichRun));
                        sb.Append("\"\"");
                        ichRun = ichCur + 1;
                    }
                }
                Contracts.Assert(ichCur == ichLim);
                if (ichRun < ichCur)
                    sb.AppendSpan(span.Slice(ichRun, ichCur - ichRun));
                if (quoted)
                    sb.Append('"');
            }
        }

        internal static void MapTimeSpan(in TimeSpan src, ref StringBuilder sb)
        {
            if (sb == null)
                sb = new StringBuilder();
            else
                sb.Clear();

            sb.AppendFormat("\"{0:c}\"", src);
        }

        internal static void MapDateTime(in DateTime src, ref StringBuilder sb)
        {
            if (sb == null)
                sb = new StringBuilder();
            else
                sb.Clear();

            sb.AppendFormat("\"{0:o}\"", src);
        }

        internal static void MapDateTimeZone(in DateTimeOffset src, ref StringBuilder sb)
        {
            if (sb == null)
                sb = new StringBuilder();
            else
                sb.Clear();

            sb.AppendFormat("\"{0:o}\"", src);
        }
    }
}

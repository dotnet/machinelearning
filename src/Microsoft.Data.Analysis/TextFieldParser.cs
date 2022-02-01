// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;

namespace Microsoft.Data.Analysis
{
    internal enum FieldType
    {
        Delimited,
        FixedWidth
    }

    internal class QuoteDelimitedFieldBuilder
    {
        private readonly StringBuilder _field;
        private bool _fieldFinished;
        private int _index;
        private int _delimiterLength;
        private readonly Regex _delimiterRegex;
        private readonly string _spaceChars;
        private bool _malformedLine;

        public QuoteDelimitedFieldBuilder(Regex delimiterRegex, string spaceChars)
        {
            _delimiterRegex = delimiterRegex;
            _spaceChars = spaceChars;
            _field = new StringBuilder();
        }

        public bool FieldFinished => _fieldFinished;

        public string Field => _field.ToString();

        public int Index => _index;

        public int DelimiterLength => _delimiterLength;

        public bool MalformedLine => _malformedLine;

        public void BuildField(string line, int startAt)
        {
            _index = startAt;
            int length = line.Length;

            while (_index < length)
            {
                if (line[_index] == '"')
                {
                    // Are we at the end of a file?
                    if (_index + 1 == length)
                    {
                        // We've found the end of the field
                        _fieldFinished = true;
                        _delimiterLength = 1;

                        // Move index past end of line
                        _index++;
                        return;
                    }
                    // Check to see if this is an escaped quote
                    if (_index + 1 < line.Length && line[_index + 1] == '"')
                    {
                        _field.Append('"');
                        _index += 2;
                        continue;
                    }

                    // Find the next delimiter and make sure everything between the quote and delimiter is ignorable
                    int Limit;
                    Match delimiterMatch = _delimiterRegex.Match(line, _index + 1);
                    if (!delimiterMatch.Success)
                    {
                        Limit = length - 1;
                    }
                    else
                    {
                        Limit = delimiterMatch.Index - 1;
                    }

                    for (int i = _index + 1; i < Limit; i++)
                    {
                        if (_spaceChars.IndexOf(line[i]) < 0)
                        {
                            _malformedLine = true;
                            return;
                        }
                    }

                    // The length of the delimiter is the length of the closing quote (1) + any spaces + the length of the delimiter we matched if any
                    _delimiterLength = 1 + Limit - _index;
                    if (delimiterMatch.Success)
                    {
                        _delimiterLength += delimiterMatch.Length;
                    }

                    _fieldFinished = true;
                    return;
                }
                else
                {
                    _field.Append(line[_index]);
                    _index += 1;
                }
            }
        }
    }


    internal class TextFieldParser : IDisposable
    {
        private delegate int ChangeBufferFunction();

        private bool _disposed;

        private TextReader _reader;

        private string[] _commentTokens = null;

        private long _lineNumber = 1;

        private bool _endOfData;

        private string _errorLine = "";

        private long _errorLineNumber = -1;

        private FieldType _textFieldType = FieldType.Delimited;

        private int[] _fieldWidths;

        private int[] _fieldWidthsCopy;

        private string[] _delimiters;

        private string[] _delimitersCopy;

        private Regex _delimiterRegex;

        private Regex _delimiterWithEndCharsRegex;

        private readonly int[] _whitespaceCodes = new int[] { '\u0009', '\u000B', '\u000C', '\u0020', '\u0085', '\u00A0', '\u1680', '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200A', '\u200B', '\u2028', '\u2029', '\u3000', '\uFEFF' };

        private Regex _beginQuotesRegex;

        private bool _trimWhiteSpace = true;

        private int _position;

        private int _peekPosition;

        private int _charsRead;

        private bool _needPropertyCheck = true;

        private const int DEFAULT_BUFFER_LENGTH = 4096;

        private char[] _buffer = new char[DEFAULT_BUFFER_LENGTH];

        private bool _hasFieldsEnclosedInQuotes = true;

        private int _lineLength;

        private string _spaceChars;

        private readonly int _maxLineSize = 10000000;

        private readonly int _maxBufferSize = 10000000;

        private readonly bool _leaveOpen;

        private readonly char[] _newLineChars = Environment.NewLine.ToCharArray();

        public string[] CommentTokens
        {
            get => _commentTokens;
            set
            {
                CheckCommentTokensForWhitespace(value);
                _commentTokens = value;
                _needPropertyCheck = true;
            }
        }

        public bool EndOfData
        {
            get
            {
                if (_endOfData)
                {
                    return _endOfData;
                }
                if ((_reader == null) || (_buffer == null))
                {
                    _endOfData = true;
                    return true;
                }
                if (PeekNextDataLine() != null)
                {
                    return false;
                }
                _endOfData = true;
                return true;
            }
        }

        public long LineNumber
        {
            get
            {
                if (_lineNumber != -1 && ((_reader.Peek() == -1) && (_position == _charsRead)))
                {
                    // Side effect of a property. Not great. Just leaving it in for now.
                    CloseReader();
                }
                return _lineNumber;
            }
        }

        public string ErrorLine => _errorLine;

        public long ErrorLineNumber => _errorLineNumber;

        public FieldType TextFieldType
        {
            get => _textFieldType;
            set
            {
                ValidateFieldTypeEnumValue(value, "value");
                _textFieldType = value;
                _needPropertyCheck = true;
            }
        }

        public int[] FieldWidths
        {
            get => _fieldWidths;
            private set
            {
                if (value != null)
                {
                    ValidateFieldWidthsOnInput(value);
                    _fieldWidthsCopy = (int[])value.Clone();
                }
                else
                {
                    _fieldWidthsCopy = null;
                }
                _fieldWidths = value;
                _needPropertyCheck = true;
            }
        }

        public string[] Delimiters
        {
            get => _delimiters;
            private set
            {
                if (value != null)
                {
                    ValidateDelimiters(value);
                    _delimitersCopy = (string[])value.Clone();
                }
                else
                {
                    _delimitersCopy = null;
                }
                _delimiters = value;
                _needPropertyCheck = true;
                _beginQuotesRegex = null;
            }
        }

        public bool TrimWhiteSpace
        {
            get => _trimWhiteSpace;
            set
            {
                _trimWhiteSpace = value;
            }
        }

        public bool HasFieldsEnclosedInQuotes
        {
            get => _hasFieldsEnclosedInQuotes;
            set
            {
                _hasFieldsEnclosedInQuotes = value;
            }
        }

        private Regex BeginQuotesRegex
        {
            get
            {
                if (_beginQuotesRegex == null)
                {
                    string pattern = string.Format(CultureInfo.InvariantCulture, "\\G[{0}]*\"", WhitespacePattern);
                    _beginQuotesRegex = new Regex(pattern, RegexOptions.CultureInvariant);
                }
                return _beginQuotesRegex;
            }
        }

        private string EndQuotePattern => string.Format(CultureInfo.InvariantCulture, "\"[{0}]*", WhitespacePattern);

        private string WhitespaceCharacters
        {
            get
            {
                StringBuilder builder = new StringBuilder();
                int[] whitespaceCodes = _whitespaceCodes;
                foreach (int code in whitespaceCodes)
                {
                    char spaceChar = (char)code;
                    if (!CharacterIsInDelimiter(spaceChar))
                    {
                        builder.Append(spaceChar);
                    }
                }
                return builder.ToString();
            }
        }

        private string WhitespacePattern
        {
            get
            {
                StringBuilder builder = new StringBuilder();
                int[] whitespaceCodes = _whitespaceCodes;
                for (int i = 0; i < whitespaceCodes.Length; i++)
                {
                    int code = whitespaceCodes[i];
                    char spaceChar = (char)code;
                    if (!CharacterIsInDelimiter(spaceChar))
                    {
                        builder.Append("\\u" + code.ToString("X4", CultureInfo.InvariantCulture));
                    }
                }
                return builder.ToString();
            }
        }

        public TextFieldParser(string path)
        {
            InitializeFromPath(path, Encoding.ASCII, detectEncoding: true);
        }

        public TextFieldParser(string path, Encoding defaultEncoding)
        {
            InitializeFromPath(path, defaultEncoding, detectEncoding: true);
        }

        public TextFieldParser(string path, Encoding defaultEncoding, bool detectEncoding)
        {
            InitializeFromPath(path, defaultEncoding, detectEncoding);
        }

        public TextFieldParser(Stream stream)
        {
            InitializeFromStream(stream, Encoding.ASCII, detectEncoding: true);
        }

        public TextFieldParser(Stream stream, Encoding defaultEncoding)
        {
            InitializeFromStream(stream, defaultEncoding, detectEncoding: true);
        }

        public TextFieldParser(Stream stream, Encoding defaultEncoding, bool detectEncoding)
        {
            InitializeFromStream(stream, defaultEncoding, detectEncoding);
        }

        public TextFieldParser(Stream stream, Encoding defaultEncoding, bool detectEncoding, bool leaveOpen)
        {
            _leaveOpen = leaveOpen;
            InitializeFromStream(stream, defaultEncoding, detectEncoding);
        }

        public TextFieldParser(TextReader reader)
        {
            _reader = reader ?? throw new ArgumentNullException(nameof(reader));
            ReadToBuffer();
        }

        public void SetDelimiters(params string[] delimiters)
        {
            Delimiters = delimiters;
        }

        public void SetFieldWidths(params int[] fieldWidths)
        {
            FieldWidths = fieldWidths;
        }


        public string ReadLine()
        {
            if ((_reader == null) || (_buffer == null))
            {
                return null;
            }

            ChangeBufferFunction BufferFunction = ReadToBuffer;
            string line = ReadNextLine(ref _position, BufferFunction);
            if (line == null)
            {
                FinishReading();
                return null;
            }

            _lineNumber++;
            return line.TrimEnd(_newLineChars);
        }

        public string[] ReadFields()
        {
            if ((_reader == null) || (_buffer == null))
            {
                return null;
            }
            ValidateReadyToRead();
            switch (_textFieldType)
            {
                case FieldType.FixedWidth:
                    return ParseFixedWidthLine();
                case FieldType.Delimited:
                    return ParseDelimitedLine();
                default:
                    Debug.Fail("The TextFieldType is not supported");
                    return null;
            }
        }

        ///<summary>
        /// Peek at <paramref name="numberOfChars"/> characters of the next data line without reading the line
        ///</summary>
        ///<param name="numberOfChars">The number of characters to look at in the next data line.</param>
        ///<returns>A string consisting of the first <paramref name="numberOfChars"/> characters of the next line. >If numberOfChars is greater than the next line, only the next line is returned</returns>
        public string PeekChars(int numberOfChars)
        {
            if (numberOfChars <= 0)
            {
                throw new ArgumentException(string.Format(Strings.PositiveNumberOfCharacters, nameof(numberOfChars)));
            }

            if ((_reader == null) || (_buffer == null))
            {
                return null;
            }

            if (_endOfData)
            {
                return null;
            }

            string line = PeekNextDataLine();
            if (line == null)
            {
                _endOfData = true;
                return null;
            }

            line = line.TrimEnd(_newLineChars);
            if (line.Length < numberOfChars)
            {
                return line;
            }

            return line.Substring(0, numberOfChars);
        }


        public string ReadToEnd()
        {
            if ((_reader == null) || (_buffer == null))
            {
                return null;
            }
            StringBuilder builder = new StringBuilder(_buffer.Length);
            builder.Append(_buffer, _position, _charsRead - _position);
            builder.Append(_reader.ReadToEnd());
            FinishReading();
            return builder.ToString();
        }

        public void Close()
        {
            CloseReader();
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (!_disposed)
                {
                    Close();
                }
                _disposed = true;
            }
        }

        private void ValidateFieldTypeEnumValue(FieldType value, string paramName)
        {
            if (value < FieldType.Delimited || value > FieldType.FixedWidth)
            {
                throw new InvalidEnumArgumentException(paramName, (int)value, typeof(FieldType));
            }
        }

        private void CloseReader()
        {
            FinishReading();
            if (_reader != null)
            {
                if (!_leaveOpen)
                {
                    _reader.Close();
                }
                _reader = null;
            }
        }

        private void FinishReading()
        {
            _lineNumber = -1L;
            _endOfData = true;
            _buffer = null;
            _delimiterRegex = null;
            _beginQuotesRegex = null;
        }

        private void InitializeFromPath(string path, Encoding defaultEncoding, bool detectEncoding)
        {
            if (path == null)
            {
                throw new ArgumentNullException(nameof(path));
            }
            if (defaultEncoding == null)
            {
                throw new ArgumentNullException(nameof(defaultEncoding));
            }
            string fullPath = ValidatePath(path);
            FileStream fileStreamTemp = new FileStream(fullPath, (FileMode.Open), (FileAccess.Read), (FileShare.ReadWrite));
            _reader = new StreamReader(fileStreamTemp, defaultEncoding, detectEncoding);
            ReadToBuffer();
        }

        private void InitializeFromStream(Stream stream, Encoding defaultEncoding, bool detectEncoding)
        {
            if (stream == null)
            {
                throw new ArgumentNullException(nameof(stream));
            }
            if (!stream.CanRead)
            {
                throw new ArgumentException(Strings.StreamDoesntSupportReading);
            }
            if (defaultEncoding == null)
            {
                throw new ArgumentNullException(nameof(defaultEncoding));
            }
            _reader = new StreamReader(stream, defaultEncoding, detectEncoding);
            ReadToBuffer();
        }

        private string ValidatePath(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException(Strings.FileNotFound);
            }
            return path;
        }

        private bool IgnoreLine(string line)
        {
            if (line == null)
            {
                return false;
            }
            string trimmedLine = line.Trim();
            if (trimmedLine.Length == 0)
            {
                return true;
            }
            if (_commentTokens != null)
            {
                string[] commentTokens = _commentTokens;
                foreach (string Token in commentTokens)
                {
                    if (Token == string.Empty)
                    {
                        continue;
                    }
                    if (trimmedLine.StartsWith(Token, StringComparison.Ordinal))
                    {
                        return true;
                    }
                    if (line.StartsWith(Token, StringComparison.Ordinal))
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        private int ReadToBuffer()
        {
            Debug.Assert(_buffer != null, "There's no buffer");
            Debug.Assert(_reader != null, "There's no StreamReader");
            _position = 0;
            int BufferLength = _buffer.Length;
            Debug.Assert(BufferLength >= DEFAULT_BUFFER_LENGTH, "Buffer shrunk to below default");
            if (BufferLength > DEFAULT_BUFFER_LENGTH)
            {
                BufferLength = DEFAULT_BUFFER_LENGTH;
                _buffer = new char[BufferLength];
            }
            _charsRead = _reader.Read(_buffer, 0, BufferLength);
            return _charsRead;
        }

        private int SlideCursorToStartOfBuffer()
        {
            Debug.Assert(_buffer != null, "There's no buffer");
            Debug.Assert(_reader != null, "There's no StreamReader");
            Debug.Assert((_position >= 0) && (_position <= _buffer.Length), "The cursor is out of range");
            if (_position > 0)
            {
                int bufferLength = _buffer.Length;
                char[] tempArray = new char[bufferLength];
                Array.Copy(_buffer, _position, tempArray, 0, bufferLength - _position);
                int charsRead = _reader.Read(tempArray, bufferLength - _position, _position);
                _charsRead = _charsRead - _position + charsRead;
                _position = 0;
                _buffer = tempArray;
                return charsRead;
            }
            return 0;
        }

        private int IncreaseBufferSize()
        {
            Debug.Assert(_buffer != null, "There's no buffer");
            Debug.Assert(_reader != null, "There's no StreamReader");
            _peekPosition = _charsRead;
            int bufferSize = _buffer.Length + DEFAULT_BUFFER_LENGTH;
            if (bufferSize > _maxBufferSize)
            {
                throw new Exception(Strings.ExceededMaxBufferSize);
            }
            char[] tempArray = new char[bufferSize];
            Array.Copy(_buffer, tempArray, _buffer.Length);
            int charsRead = _reader.Read(tempArray, _buffer.Length, DEFAULT_BUFFER_LENGTH);
            _buffer = tempArray;
            _charsRead += charsRead;
            Debug.Assert(_charsRead <= bufferSize, "We've read more chars than we have space for");
            return charsRead;
        }

        private string ReadNextDataLine()
        {
            ChangeBufferFunction BufferFunction = ReadToBuffer;
            string line;
            do
            {
                line = ReadNextLine(ref _position, BufferFunction);
                _lineNumber++;
            }
            while (IgnoreLine(line));
            if (line == null)
            {
                CloseReader();
            }
            return line;
        }

        private string PeekNextDataLine()
        {
            ChangeBufferFunction BufferFunction = IncreaseBufferSize;
            SlideCursorToStartOfBuffer();
            _peekPosition = 0;
            string line;
            do
            {
                line = ReadNextLine(ref _peekPosition, BufferFunction);
            }
            while (IgnoreLine(line));
            return line;
        }

        private string ReadNextLine(ref int cursor, ChangeBufferFunction changeBuffer)
        {
            Debug.Assert(_buffer != null, "There's no buffer");
            Debug.Assert((cursor >= 0) && (cursor <= _charsRead), "The cursor is out of range");
            if (cursor == _charsRead && changeBuffer() == 0)
            {
                return null;
            }
            StringBuilder Builder = null;
            // Consider replacing this do-while with a string search to take advantage of vectorization
            do
            {
                for (int i = cursor; i <= _charsRead - 1; i++)
                {
                    char Character = _buffer[i];
                    if (!(Character.Equals('\r') || Character.Equals('\n')))
                    {
                        continue;
                    }
                    if (Builder != null)
                    {
                        Builder.Append(_buffer, cursor, i - cursor + 1);
                    }
                    else
                    {
                        Builder = new StringBuilder(i + 1);
                        Builder.Append(_buffer, cursor, i - cursor + 1);
                    }
                    cursor = i + 1;
                    if (Character.Equals('\r'))
                    {
                        if (cursor < _charsRead)
                        {
                            if (_buffer[cursor].Equals('\n'))
                            {
                                cursor++;
                                Builder.Append("\n");
                            }
                        }
                        else if (changeBuffer() > 0 && _buffer[cursor].Equals('\n'))
                        {
                            cursor++;
                            Builder.Append("\n");
                        }
                    }
                    return Builder.ToString();
                }

                // Searched the whole buffer and haven't found an end of line. Save what we have and read more to the buffer
                int Size = _charsRead - cursor;
                if (Builder == null)
                {
                    Builder = new StringBuilder(Size + 10);
                }
                Builder.Append(_buffer, cursor, Size);
            }
            while (changeBuffer() > 0);

            return Builder.ToString();
        }

        private string[] ParseDelimitedLine()
        {
            string line = ReadNextDataLine();
            if (line == null)
            {
                return null;
            }
            long currentLineNumber = _lineNumber - 1;
            int index = 0;
            List<string> Fields = new List<string>();
            int lineEndIndex = GetEndOfLineIndex(line);
            while (index <= lineEndIndex)
            {
                Match matchResult = null;
                bool quoteDelimited = false;
                if (HasFieldsEnclosedInQuotes)
                {
                    matchResult = BeginQuotesRegex.Match(line, index);
                    quoteDelimited = matchResult.Success;
                }
                string field;
                if (quoteDelimited)
                {
                    // Move the Index beyond quote
                    index = matchResult.Index + matchResult.Length;

                    // Looking for the closing quote
                    QuoteDelimitedFieldBuilder endHelper = new QuoteDelimitedFieldBuilder(_delimiterWithEndCharsRegex, _spaceChars);
                    endHelper.BuildField(line, index);
                    if (endHelper.MalformedLine)
                    {
                        _errorLine = line.TrimEnd(_newLineChars);
                        _errorLineNumber = currentLineNumber;
                        throw new Exception(string.Format(Strings.CannotParseWithDelimiters, currentLineNumber));
                    }
                    if (endHelper.FieldFinished)
                    {
                        field = endHelper.Field;
                        index = endHelper.Index + endHelper.DelimiterLength;
                    }
                    else
                    {
                        // We may have an embedded line end character, so grab next line
                        do
                        {
                            int endOfLine = line.Length;
                            string newLine = ReadNextDataLine();
                            if (newLine == null)
                            {
                                _errorLine = line.TrimEnd(_newLineChars);
                                _errorLineNumber = currentLineNumber;
                                throw new Exception(string.Format(Strings.CannotParseWithDelimiters, currentLineNumber));
                            }
                            if (line.Length + newLine.Length > _maxLineSize)
                            {
                                _errorLine = line.TrimEnd(_newLineChars);
                                _errorLineNumber = currentLineNumber;
                                throw new Exception(string.Format(Strings.LineExceedsMaxLineSize, currentLineNumber));
                            }
                            line += newLine;
                            lineEndIndex = GetEndOfLineIndex(line);
                            endHelper.BuildField(line, endOfLine);
                            if (endHelper.MalformedLine)
                            {
                                _errorLine = line.TrimEnd(_newLineChars);
                                _errorLineNumber = currentLineNumber;
                                throw new Exception(string.Format(Strings.CannotParseWithDelimiters, currentLineNumber));
                            }
                        }
                        while (!endHelper.FieldFinished);
                        field = endHelper.Field;
                        index = endHelper.Index + endHelper.DelimiterLength;
                    }
                    if (_trimWhiteSpace)
                    {
                        field = field.Trim();
                    }
                    Fields.Add(field);
                    continue;
                }

                // Find the next delimiter
                Match delimiterMatch = _delimiterRegex.Match(line, index);
                if (delimiterMatch.Success)
                {
                    field = line.Substring(index, delimiterMatch.Index - index);
                    if (_trimWhiteSpace)
                    {
                        field = field.Trim();
                    }
                    Fields.Add(field);
                    index = delimiterMatch.Index + delimiterMatch.Length;
                    continue;
                }
                field = line.Substring(index).TrimEnd(_newLineChars);
                if (_trimWhiteSpace)
                {
                    field = field.Trim();
                }
                Fields.Add(field);
                break;
            }
            return Fields.ToArray();
        }

        private string[] ParseFixedWidthLine()
        {
            Debug.Assert(_fieldWidths != null, "No field widths");
            string line = ReadNextDataLine();
            if (line == null)
            {
                return null;
            }
            line = line.TrimEnd(_newLineChars);
            StringInfo lineInfo = new StringInfo(line);
            ValidateFixedWidthLine(lineInfo, _lineNumber - 1);
            int index = 0;
            int length = _fieldWidths.Length;
            string[] Fields = new string[length];
            for (int i = 0; i < length; i++)
            {
                Fields[i] = GetFixedWidthField(lineInfo, index, _fieldWidths[i]);
                index += _fieldWidths[i];
            }
            return Fields;
        }

        private string GetFixedWidthField(StringInfo line, int index, int fieldLength)
        {
            string field = (fieldLength > 0) ? line.SubstringByTextElements(index, fieldLength) : ((index < line.LengthInTextElements) ? line.SubstringByTextElements(index).TrimEnd(_newLineChars) : string.Empty);
            if (_trimWhiteSpace)
            {
                return field.Trim();
            }
            return field;
        }

        private int GetEndOfLineIndex(string line)
        {
            Debug.Assert(line != null, "We are parsing null");
            int length = line.Length;
            Debug.Assert(length > 0, "A blank line shouldn't be parsed");
            if (length == 1)
            {
                Debug.Assert(!line[0].Equals('\r') && !line[0].Equals('\n'), "A blank line shouldn't be parsed");
                return length;
            }
            checked
            {
                if (line[length - 2].Equals('\r') || line[length - 2].Equals('\n'))
                {
                    return length - 2;
                }
                if (line[length - 1].Equals('\r') || line[length - 1].Equals('\n'))
                {
                    return length - 1;
                }
                return length;
            }
        }

        private void ValidateFixedWidthLine(StringInfo line, long lineNumber)
        {
            Debug.Assert(line != null, "No Line sent");
            if (line.LengthInTextElements < _lineLength)
            {
                _errorLine = line.String;
                _errorLineNumber = checked(_lineNumber - 1);
                throw new Exception(string.Format(Strings.CannotParseWithFieldWidths, lineNumber));
            }
        }

        private void ValidateFieldWidths()
        {
            if (_fieldWidths == null)
            {
                throw new InvalidOperationException(Strings.NullFieldWidths);
            }
            if (_fieldWidths.Length == 0)
            {
                throw new InvalidOperationException(Strings.EmptyFieldWidths);
            }
            checked
            {
                int widthBound = _fieldWidths.Length - 1;
                _lineLength = 0;
                int num = widthBound - 1;
                for (int i = 0; i <= num; i++)
                {
                    Debug.Assert(_fieldWidths[i] > 0, "Bad field width, this should have been caught on input");
                    _lineLength += _fieldWidths[i];
                }
                if (_fieldWidths[widthBound] > 0)
                {
                    _lineLength += _fieldWidths[widthBound];
                }
            }
        }

        private void ValidateFieldWidthsOnInput(int[] widths)
        {
            Debug.Assert(widths != null, "There are no field widths");
            int bound = widths.Length - 1;
            for (int i = 0; i <= bound - 1; i++)
            {
                if (widths[i] < 1)
                {
                    throw new ArgumentException(Strings.InvalidFieldWidths);
                }
            }
        }

        private void ValidateAndEscapeDelimiters()
        {
            if (_delimiters == null)
            {
                throw new Exception(Strings.NullDelimiters);
            }
            if (_delimiters.Length == 0)
            {
                throw new Exception(Strings.EmptyDelimiters);
            }
            int length = _delimiters.Length;
            StringBuilder builder = new StringBuilder();
            StringBuilder quoteBuilder = new StringBuilder();
            quoteBuilder.Append(EndQuotePattern + "(");
            for (int i = 0; i <= length - 1; i++)
            {
                if (_delimiters[i] != null)
                {
                    if (_hasFieldsEnclosedInQuotes && _delimiters[i].IndexOf('"') > -1)
                    {
                        throw new Exception(Strings.IllegalQuoteDelimiter);
                    }
                    string escapedDelimiter = Regex.Escape(_delimiters[i]);
                    builder.Append(escapedDelimiter + "|");
                    quoteBuilder.Append(escapedDelimiter + "|");
                }
                else
                {
                    Debug.Fail("Delimiter element is empty. This should have been caught on input");
                }
            }
            _spaceChars = WhitespaceCharacters;
            _delimiterRegex = new Regex(builder.ToString(0, builder.Length - 1), (RegexOptions)512);
            builder.Append("\r|\n");
            _delimiterWithEndCharsRegex = new Regex(builder.ToString(), (RegexOptions)512);
            quoteBuilder.Append("\r|\n)|\"$");
        }

        private void ValidateReadyToRead()
        {
            if (!(_needPropertyCheck || ArrayHasChanged()))
            {
                return;
            }
            switch (_textFieldType)
            {
                case FieldType.Delimited:
                    ValidateAndEscapeDelimiters();
                    break;
                case FieldType.FixedWidth:
                    ValidateFieldWidths();
                    break;
                default:
                    Debug.Fail("Unknown TextFieldType");
                    break;
            }
            if (_commentTokens != null)
            {
                string[] commentTokens = _commentTokens;
                foreach (string token in commentTokens)
                {
                    if (token != string.Empty && (_hasFieldsEnclosedInQuotes && (_textFieldType == FieldType.Delimited)) && string.Compare(token.Trim(), "\"", StringComparison.Ordinal) == 0)
                    {
                        throw new Exception(Strings.IllegalQuoteDelimiter);
                    }
                }
            }
            _needPropertyCheck = false;
        }

        private void ValidateDelimiters(string[] delimiterArray)
        {
            if (delimiterArray == null)
            {
                return;
            }
            foreach (string delimiter in delimiterArray)
            {
                if (delimiter == string.Empty)
                {
                    throw new Exception(Strings.EmptyDelimiters);
                }
                if (delimiter.IndexOfAny(_newLineChars) > -1)
                {
                    throw new Exception(Strings.DelimiterCannotBeNewlineChar);
                }
            }
        }

        private bool ArrayHasChanged()
        {
            int lowerBound = 0;
            int upperBound = 0;
            switch (_textFieldType)
            {
                case FieldType.Delimited:
                    {
                        Debug.Assert(((_delimitersCopy == null) && (_delimiters == null)) || ((_delimitersCopy != null) && (_delimiters != null)), "Delimiters and copy are not both Nothing or both not Nothing");
                        if (_delimiters == null)
                        {
                            return false;
                        }
                        lowerBound = _delimitersCopy.GetLowerBound(0);
                        upperBound = _delimitersCopy.GetUpperBound(0);
                        int num3 = lowerBound;
                        int num4 = upperBound;
                        for (int i = num3; i <= num4; i++)
                        {
                            if (_delimiters[i] != _delimitersCopy[i])
                            {
                                return true;
                            }
                        }
                        break;
                    }
                case FieldType.FixedWidth:
                    {
                        Debug.Assert(((_fieldWidthsCopy == null) && (_fieldWidths == null)) || ((_fieldWidthsCopy != null) && (_fieldWidths != null)), "FieldWidths and copy are not both Nothing or both not Nothing");
                        if (_fieldWidths == null)
                        {
                            return false;
                        }
                        lowerBound = _fieldWidthsCopy.GetLowerBound(0);
                        upperBound = _fieldWidthsCopy.GetUpperBound(0);
                        int num = lowerBound;
                        int num2 = upperBound;
                        for (int j = num; j <= num2; j++)
                        {
                            if (_fieldWidths[j] != _fieldWidthsCopy[j])
                            {
                                return true;
                            }
                        }
                        break;
                    }
                default:
                    Debug.Fail("Unknown TextFieldType");
                    break;
            }
            return false;
        }

        private void CheckCommentTokensForWhitespace(string[] tokens)
        {
            if (tokens == null)
            {
                return;
            }
            foreach (string token in tokens)
            {
                if (token.Length == 1 && char.IsWhiteSpace(token[0]))
                {
                    throw new Exception(Strings.CommentTokenCannotContainWhitespace);
                }
            }
        }

        private bool CharacterIsInDelimiter(char testCharacter)
        {
            Debug.Assert(_delimiters != null, "No delimiters set!");
            string[] delimiters = _delimiters;
            foreach (string delimiter in delimiters)
            {
                if (delimiter.IndexOf(testCharacter) > -1)
                {
                    return true;
                }
            }
            return false;
        }
    }
}

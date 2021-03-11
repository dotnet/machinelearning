// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
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

        private Regex _whiteSpaceRegEx = new Regex("\\s", RegexOptions.CultureInvariant);

        private bool _trimWhiteSpace = true;

        private int _position;

        private int _peekPosition;

        private int _charsRead;

        private const int DEFAULT_BUFFER_LENGTH = 4096;

        private char[] _buffer = new char[DEFAULT_BUFFER_LENGTH];

        private bool _hasFieldsEnclosedInQuotes = true;

        private int _maxBufferSize = 10000000;

        private bool _leaveOpen;

        private char[] newLineChars = Environment.NewLine.ToCharArray();

        public string[] CommentTokens
        {
            get => _commentTokens;
            set
            {
                CheckCommentTokensForWhitespace(value);
                _commentTokens = value;
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
                if ((_reader == null) | (_buffer == null))
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
                if (_lineNumber != -1 && ((_reader.Peek() == -1) & (_position == _charsRead)))
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
            get =>_textFieldType;
            set
            {
                ValidateFieldTypeEnumValue(value, "value");
                _textFieldType = value;
            }
        }

        public int[] FieldWidths
        {
            get =>_fieldWidths;
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
            }
        }

        public bool TrimWhiteSpace
        {
            get =>_trimWhiteSpace;
            set
            {
                _trimWhiteSpace = value;
            }
        }

        public bool HasFieldsEnclosedInQuotes
        {
            get =>_hasFieldsEnclosedInQuotes;
            set
            {
                _hasFieldsEnclosedInQuotes = value;
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
            if ((_reader == null) | (_buffer == null))
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
            return line.TrimEnd(newLineChars);
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
                throw new ArgumentException($"{nameof(numberOfChars)} must be greater than 0");
            }

            if ((_reader == null) | (_buffer == null))
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

            line = line.TrimEnd(newLineChars);
            if (line.Length < numberOfChars)
            {
                return line;
            }

            return line.Substring(0, numberOfChars);
        }


        public string ReadToEnd()
        {
            if ((_reader == null) | (_buffer == null))
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
                throw new ArgumentException("stream can't read");
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
                throw new FileNotFoundException($"{path} not found.");
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
            Debug.Assert((_position >= 0) & (_position <= _buffer.Length), "The cursor is out of range");
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
                throw new Exception("Exceeded maximum buffer size");
            }
            char[] tempArray = new char[bufferSize];
            Array.Copy(_buffer, tempArray, _buffer.Length);
            int charsRead = _reader.Read(tempArray, _buffer.Length, DEFAULT_BUFFER_LENGTH);
            _buffer = tempArray;
            _charsRead += charsRead;
            Debug.Assert(_charsRead <= bufferSize, "We've read more chars than we have space for");
            return charsRead;
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
            Debug.Assert((cursor >= 0) & (cursor <= _charsRead), "The cursor is out of range");
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
                    if (!(Character.Equals('\r') | Character.Equals('\n')))
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

        private void ValidateFieldWidthsOnInput(int[] widths)
        {
            Debug.Assert(widths != null, "There are no field widths");
            int bound = widths.Length - 1;
            for (int i = 0; i <= bound - 1; i++)
            {
                if (widths[i] < 1)
                {
                    throw new ArgumentException("All field widths, except the last element, must be greater than zero. A field width less than or equal to zero in the last element indicates the last field is of variable length.");
                }
            }
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
                    throw new Exception("Delimiter cannot be empty");
                }
                if (delimiter.IndexOfAny(newLineChars) > -1)
                {
                    throw new Exception("Delimiter cannot be new line characters");
                }
            }
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
                    throw new Exception("Comment token cannot contain whitespace");
                }
            }
        }
    }
}

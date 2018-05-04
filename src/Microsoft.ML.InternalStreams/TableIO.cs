// owner: rragno

using System;
using System.IO;
using System.Collections;
using System.Collections.Specialized;
using System.Text;
#if !ENABLE_BARTOK
using System.Xml;
#endif
#if ALLOW_DB
using System.Data;
using System.Data.OleDb;
#endif

namespace Microsoft.ML.Runtime.Internal.IO
{
    //// TODO: consider unifying all formats, syncing with other data access models
    //// TODO: simple XML support
    //// TODO: consider generalizing header support; changing from "row,item" to "item,field";
    ////       making a base TableReader abstract class.

    /// <summary>
    /// Process tabular data.
    /// </summary>
    public interface ITableProcessor
    {
        /// <summary>
        /// Gets or sets whether to trim whitespace from each field.
        /// </summary>
        bool TrimWhitespace { get; set; }

        /// <summary>
        /// Gets or sets whether to ignore case when matching header names.
        /// </summary>
        bool IgnoreHeaderCase { get; set; }

        /// <summary>
        /// Gets or sets the header names.
        /// </summary>
        string[] Headers { get; set; }

        /// <summary>
        /// Check for end of file.
        /// </summary>
        /// <returns>true if at end of file, false otherwise</returns>
        bool Eof();

        /// <summary>
        /// Advance to the next row.
        /// </summary>
        void NextRow();

        /// <summary>
        /// Close the table.
        /// </summary>
        void Close();
    }

    /// <summary>
    ///
    /// </summary>
    public interface ITableRow
    {
        /// <summary>
        /// Get the field at the column index.
        /// </summary>
        string this[int index] { get; }

        /// <summary>
        /// Get the field at the column with the given header.
        /// </summary>
        string this[string header] { get; }
    }

    /// <summary>
    /// Read tabular data.
    /// </summary>
    public interface ITableReader : ITableProcessor, IEnumerable, ITableRow
    {
        /// <summary>
        /// Gets or sets whether to return "", not null, when the end of a row is reached,
        /// until the row is advanced:
        /// </summary>
        bool FillBlankColumns { get; set; }

        /// <summary>
        /// Check for end of row.
        /// </summary>
        /// <returns>true if at end of row, false otherwise</returns>
        bool RowEnd();

        /// <summary>
        /// Get the next field and advance the reader.
        /// </summary>
        /// <returns>the field at the next column</returns>
        string ReadItem();

        /// <summary>
        /// Get the next field and advance the reader, filling with empty fields at the end of the row.
        /// </summary>
        /// <returns>the field at the next column</returns>
        string ReadItemLinear();

        /// <summary>
        /// Get the number of fields in the current row.
        /// </summary>
        /// <returns>the number of fields in the current row</returns>
        int RowLength();

        /// <summary>
        /// Read an entire row and advance the reader.
        /// </summary>
        /// <returns>The current row as an array of fields</returns>
        string[] ReadRow();

        /// <summary>
        /// Read an entire row and advance the reader.
        /// </summary>
        /// <param name="len">The length of the row to read, truncating or filling with empty fields as needed</param>
        /// <returns>The current row as an array of fields</returns>
        string[] ReadRow(int len);

        /*
        /// <summary>
        /// Get the field at the column index.
        /// </summary>
        string   this[int index]  { get; }

        /// <summary>
        /// Get the field at the column with the given header.
        /// </summary>
        string   this[string header]  { get; }
        */

        /// <summary>
        /// Reset the reader to the beginning.
        /// </summary>
        void Reset();
    }

    /// <summary>
    /// Write tabular data.
    /// </summary>
    public interface ITableWriter : ITableProcessor
    {
        /// <summary>
        /// Write the next field and advance the writer.
        /// </summary>
        /// <param name="item">the field to write</param>
        void WriteItem(string item);

        /// <summary>
        /// Write an entire row and advance the writer.
        /// </summary>
        /// <param name="items">The row to write as an array of fields</param>
        void WriteRow(string[] items);

        /// <summary>
        /// Write an entire row and advance the writer.
        /// </summary>
        /// <param name="len">The length of the row to write, truncating or filling with empty fields as needed</param>
        /// <param name="items">The row to write as an array of fields</param>
        void WriteRow(string[] items, int len);

        /// <summary>
        /// Set the field at the column index.
        /// </summary>
        string this[int index] { set; }

        /// <summary>
        /// Set the field at the column with the given header.
        /// </summary>
        string this[string header] { set; }
    }

    /// <summary>
    /// Enumerator to read through the rows in a table.
    /// </summary>
    public class TableEnumerator : IEnumerator
    {
        ITableReader _reader;

        #region IEnumerator Members
        /// <summary>
        /// Create a new enumerator to read through the table rows
        /// </summary>
        /// <param name="reader">the table to read lines from</param>
        public TableEnumerator(ITableReader reader)
        {
            _reader = reader;
        }
        /// <summary>
        /// Return the enumerator to the initial state.
        /// </summary>
        public void Reset()
        {
            _reader.Reset();
        }
        /// <summary>
        /// Get the current row of the table.
        /// </summary>
        public ITableRow Current
        {
            get
            {
                return (ITableRow)_reader;
            }
        }

        /// <summary>
        /// Get the current row of the table.
        /// </summary>
        object IEnumerator.Current
        {
            get
            {
                return ((TableEnumerator)this).Current;
            }
        }

        /// <summary>
        /// Move the enumerator to the next row.
        /// </summary>
        /// <returns>true if the next row exists, or false if at the end of the table</returns>
        public bool MoveNext()
        {
            _reader.NextRow();
            return !_reader.Eof();
        }

        #endregion
    }

    //////////////////////////////////////////////
    ////// CSV support
    //////////////////////////////////////////////

    /// <summary>
    /// Read TSV formatted data.
    /// </summary>
    public class TsvReader : CsvReader
    {
        private void Configure()
        {
            Delimiter = "\t";
            DelimiterSet = false;
            ParseQuotes = false;
            ReadHeaders = true;
            SkipBlankColumnsLines = true;
            SkipBlankLines = true;
            // should this one be set? ***
            IgnoreHeaderCase = true;
        }

        /// <summary>
        /// Create a TsvReader based on the TextReader,
        /// </summary>
        /// <param name="tr">the TextReader to read the table from</param>
        public TsvReader(TextReader tr)
            : base(tr)
        {
            Configure();
        }

        /// <summary>
        /// Create a TsvReader based on the specified file,
        /// </summary>
        /// <param name="fname">the name of the file to read the table from</param>
        public TsvReader(string fname)
            : base(fname)
        {
            Configure();
        }

        /// <summary>
        /// Create a TsvReader based on the Stream,
        /// </summary>
        /// <param name="fstream">the Stream to read the table from</param>
        public TsvReader(Stream fstream)
            : base(fstream)
        {
            Configure();
        }

        /// <summary>
        /// Create a TsvReader based on the specified file,
        /// </summary>
        /// <param name="fname">the name of the file to read the table from</param>
        /// <param name="encoding">the encoding to use to interpet the file</param>
        public TsvReader(string fname, Encoding encoding)
            : base(fname, encoding)
        {
            Configure();
        }

        /// <summary>
        /// Create a TsvReader based on the Stream,
        /// </summary>
        /// <param name="fstream">the Stream to read the table from</param>
        /// <param name="encoding">the encoding to use to interpet the Stream</param>
        public TsvReader(Stream fstream, Encoding encoding)
            : base(fstream, encoding)
        {
            Configure();
        }
    }

    /// <summary>
    /// Tab-Separated Value writer.
    /// </summary>
    public class TsvWriter : CsvWriter
    {
        private void Configure()
        {
            Delimiter = "\t";
            ParseQuotes = false;
            SkipBlankLines = true;
            EndInNewline = true;
            // should this one be set? ***
            IgnoreHeaderCase = true;
        }

        /// <summary>
        /// Create a TsvWriter based on the TextWriter,
        /// </summary>
        /// <param name="tr">the TextWriter to write the table to</param>
        public TsvWriter(TextWriter tr)
            : base(tr)
        {
            Configure();
        }

        /// <summary>
        /// Create a TsvWriter based on the specified file,
        /// </summary>
        /// <param name="fname">the name of the file to write the table to</param>
        public TsvWriter(string fname)
            : base(fname)
        {
            Configure();
        }

        /// <summary>
        /// Create a TsvWriter based on the Stream,
        /// </summary>
        /// <param name="fstream">the Stream to write the table to</param>
        public TsvWriter(Stream fstream)
            : base(fstream)
        {
            Configure();
        }

        /// <summary>
        /// Create a TsvWriter based on the specified file,
        /// </summary>
        /// <param name="fname">the name of the file to write the table to</param>
        /// <param name="encoding">the encoding to use</param>
        public TsvWriter(string fname, Encoding encoding)
            : base(fname, encoding)
        {
            Configure();
        }

        /// <summary>
        /// Create a TsvWriter based on the Stream,
        /// </summary>
        /// <param name="fstream">the Stream to write the table to</param>
        /// <param name="encoding">the encoding to use</param>
        public TsvWriter(Stream fstream, Encoding encoding)
            : base(fstream, encoding)
        {
            Configure();
        }
    }

    /// <summary>
    /// Read CSV formatted data.
    /// </summary>
    public class CsvReader : ITableReader, IDisposable
    {
        // trim whitespace from each entry:
        private bool _trimWhitespace = true;
        // skip lines that are only whitespace:
        private bool _skipBlankLines = true;
        // skip lines that have delimeters but only whitespace otherwise:
        private bool _skipBlankColumnsLines = true;
        // return "", not null, when the end of a row is reached, until the row is advanced:
        private bool _fillBlankColumns = true;
        // treat repeated delimiters as a single delimiter:
        private bool _collapseDelimiters = false;
        // treat delimiter string as a set of delimiter characters:
        private bool _delimiterSet = false;
        // read the first line as header names for the columns
        private bool _readHeaders = false;
        // the names of the headers
        private string[] _headers = null;
        private string[] _headersNormalized = null;
        private bool _ignoreHeaderCase = true;
        private bool _initialized = false;
        // use the quoteChar to determine quoted sections:
        private bool _parseQuotes = true;
        private string _quoteChar = "\"";
        private string _delimiter = ",";

        private TextReader _file;
        private string _curLine;
        private long _rowNumber = 0;  // gives 1-based rows, like excel...

        // support 1-row read-ahead:
        private StringCollection _curRow;
        private string[] _curRowArray;
        private int _curCol;

        /// <summary>
        /// Create a CsvReader based on the TextReader,
        /// </summary>
        /// <param name="tr">the TextReader to read the table from</param>
        public CsvReader(TextReader tr)
        {
            _file = tr;
            //if (file == null)
            //	Console.Out.WriteLine("  Error: CsvReader could not open null file!");
        }

        /// <summary>
        /// Create a CsvReader based on the specified file,
        /// </summary>
        /// <param name="fname">the name of the file to read the table from</param>
        public CsvReader(string fname)
            : this(ZStreamReader.Open(fname))
        //			: this(new StreamReader(fname, Encoding.UTF8, true))
        {
        }

        /// <summary>
        /// Create a CsvReader based on the Stream,
        /// </summary>
        /// <param name="fstream">the Stream to read the table from</param>
        public CsvReader(Stream fstream)
            : this(new StreamReader(fstream))
        {
        }

        /// <summary>
        /// Create a CsvReader based on the specified file,
        /// </summary>
        /// <param name="fname">the name of the file to read the table from</param>
        /// <param name="encoding">the encoding to use to interpet the file</param>
        public CsvReader(string fname, Encoding encoding)
            //			: this(ZStreamReader.Open(fname, encoding))
            : this(new StreamReader(fname, encoding, true))
        {
        }

        /// <summary>
        /// Create a CsvReader based on the Stream,
        /// </summary>
        /// <param name="fstream">the Stream to read the table from</param>
        /// <param name="encoding">the encoding to use to interpet the Stream</param>
        public CsvReader(Stream fstream, Encoding encoding)
            : this(new StreamReader(fstream, encoding, true))
        {
        }

        /// <summary>
        /// Gets or sets whether to trim whitespace from each field.
        /// </summary>
        public bool TrimWhitespace
        {
            get { return _trimWhitespace; }
            set { _trimWhitespace = value; }
        }
        /// <summary>
        /// Get or set whether to skip blank lines.
        /// </summary>
        public bool SkipBlankLines
        {
            get { return _skipBlankLines; }
            set { _skipBlankLines = value; }
        }
        /// <summary>
        /// Get or set whether to skip lines with all fields empty.
        /// </summary>
        public bool SkipBlankColumnsLines
        {
            get { return _skipBlankColumnsLines; }
            set { _skipBlankColumnsLines = value; }
        }
        /// <summary>
        /// Gets or sets whether to return "", not null, when the end of a row is reached,
        /// until the row is advanced:
        /// </summary>
        public bool FillBlankColumns
        {
            get { return _fillBlankColumns; }
            set { _fillBlankColumns = value; }
        }
        /// <summary>
        /// Get or set whether to respect quotes when parsing
        /// </summary>
        public bool ParseQuotes
        {
            get { return _quoteChar != null && _quoteChar.Length > 0 && _parseQuotes; }
            set { _parseQuotes = value; }
        }
        /// <summary>
        /// Get or set the string to use for a quote symbol
        /// </summary>
        public string QuoteChar
        {
            get { return _quoteChar; }
            set { _quoteChar = value; }
        }
        /// <summary>
        /// Get or set the column delimiter string.
        /// </summary>
        public string Delimiter
        {
            get { return _delimiter; }
            set { _delimiter = value; }
        }
        /// <summary>
        /// Get or set whether to collapse consecutive delimiters.
        /// </summary>
        public bool CollapseDelimiters
        {
            get { return _collapseDelimiters; }
            set { _collapseDelimiters = value; }
        }
        /// <summary>
        /// Get or set whether to treate the delimiter string as a set of characters.
        /// </summary>
        public bool DelimiterSet
        {
            get { return _delimiterSet; }
            set { _delimiterSet = value; }
        }
        /// <summary>
        /// Get or set whether to read the headers from the first line of the input.
        /// </summary>
        public bool ReadHeaders
        {
            get { return _readHeaders; }
            set { _readHeaders = value; }
        }
        /// <summary>
        /// Gets or sets whether to ignore case when matching header names.
        /// </summary>
        public bool IgnoreHeaderCase
        {
            get { return _ignoreHeaderCase; }
            set
            {
                if (_ignoreHeaderCase != value)
                {
                    _ignoreHeaderCase = value;
                    FixupNormalizedHeaders();
                }
            }
        }
        /// <summary>
        /// Gets or sets the header names.
        /// </summary>
        public string[] Headers
        {
            get
            {
                Initialize();
                return _headers;
            }
            set
            {
                _headers = value;
                FixupNormalizedHeaders();
            }
        }
        /// <summary>
        /// Get the number of the current row.
        /// </summary>
        public long RowNumber
        {
            get { return _rowNumber; }
        }

        /// <summary>
        /// Check for end of file.
        /// </summary>
        /// <returns>true if at end of file, false otherwise</returns>
        public bool Eof()
        {
            // a little tricky when finishing the last row with remaining whitespace...
            // won't show as EOF until after nextRow is called...
            if (_file == null)
                return true;
            if (_file.Peek() != -1)
                return false;
            if (!_initialized)
                Initialize();
            return RowEnd();
        }

        /// <summary>
        /// Check for end of row.
        /// </summary>
        /// <returns>true if at end of row, false otherwise</returns>
        public bool RowEnd()
        {
            return ((_curRow == null || _curCol >= _curRow.Count) &&
                (_curRowArray == null || _curCol >= _curRowArray.Length));
        }

        private void Initialize()
        {
            if (_initialized)
                return;
            _initialized = true;
            NextRow();
            // read and store the column header names:
            if (ReadHeaders)
            {
                // always skip blanks when reading headers
                bool givenSkipBlankLines = SkipBlankLines;
                bool givenSkipBlankColumnsLines = SkipBlankColumnsLines;
                SkipBlankLines = true;
                SkipBlankColumnsLines = true;
                Headers = ReadRow();
                SkipBlankLines = givenSkipBlankLines;
                SkipBlankColumnsLines = givenSkipBlankColumnsLines;
            }
        }

        private void FixupNormalizedHeaders()
        {
            if (_headers == null)
            {
                _headersNormalized = null;
                return;
            }
            _headersNormalized = new string[_headers.Length];
            for (int i = 0; i < _headers.Length; i++)
            {
                string header = _headers[i];
                if (header == null)
                    continue;
                header = header.Trim();
                if (IgnoreHeaderCase)
                    header = header.ToLower();
                _headersNormalized[i] = header;
            }
        }

        /// <summary>
        /// Get the field at the column index.
        /// </summary>
        public string this[int index]
        {
            get
            {
                if (!_initialized)
                    Initialize();
                if (_curRow == null)
                {
                    // look in the string[], instead...
                    if (_curRowArray == null)
                        return null;
                    if (index < 0 || index >= _curRowArray.Length)
                    {
                        if (_fillBlankColumns)
                            return "";
                        return null;
                    }
                    return _curRowArray[index];
                }
                if (index < 0 || index >= _curRow.Count)
                {
                    if (_fillBlankColumns)
                        return "";
                    return null;
                }
                return _curRow[index];
            }
        }

        /// <summary>
        /// Get the field at the column with the given header.
        /// </summary>
        public string this[string header]
        {
            get
            {
                if (!_initialized)
                    Initialize();
                if (header == null || _headers == null)
                    return null;
                header = header.Trim();
                if (IgnoreHeaderCase)
                    header = header.ToLower();
                for (int i = 0; i < _headers.Length; i++)
                {
                    if (header == _headersNormalized[i])
                    {
                        return this[i];
                    }
                }
                return null;
            }
        }

        /// <summary>
        /// Get the index of the column which has the given header.
        /// </summary>
        public int GetColumnIndex(string header)
        {
            if (!_initialized)
                Initialize();
            if (header == null || _headers == null)
                return -1;
            header = header.Trim();
            if (IgnoreHeaderCase)
                header = header.ToLower();
            for (int i = 0; i < _headers.Length; i++)
            {
                if (header == _headersNormalized[i])
                {
                    return i;
                }
            }
            return -1;
        }

        /// <summary>
        /// Get the next field and advance the reader.
        /// </summary>
        /// <returns>the field at the next column</returns>
        public string ReadItem()
        {
            // check if we need to setup the reading:
            if (!_initialized)
                Initialize();

            //// Should actually return a blank entry for the *last* blank entry and null beyond ***
            if (RowEnd())
            {
                if (_fillBlankColumns)
                    return "";
                else
                    return null;
            }

            string res = _curRow != null ? _curRow[_curCol] : _curRowArray[_curCol];
            _curCol++;
            return res;
        }

        // should optimize for 1-character quotes and delimiters...
        private string ReadItemFromLine(ref int curPos)
        {
            if (curPos >= _curLine.Length)
                return null;

            string resStr;
            curPos++;
            // check for empty case:
            bool empty = (curPos >= _curLine.Length);
            if (!empty)
            {
                // optimized check for empty in middle of line
                if (_delimiterSet)  // look for *any* of the characters
                {
                    if (_curLine.Length - curPos >= 1 &&
                        _delimiter.IndexOf(_curLine[curPos]) >= 0)
                    {
                        // leave cursor on the delimiter
                        empty = true;
                    }
                }
                else
                {
                    if (_curLine.Length - curPos >= _delimiter.Length &&
                        string.CompareOrdinal(_curLine, curPos, _delimiter, 0, _delimiter.Length) == 0)
                    {
                        // leave cursor on end of the delimiter:
                        curPos += _delimiter.Length - 1;
                        empty = true;
                    }
                }
            }
            if (empty)
            {
                resStr = "";
            }
            else
            {
                // check for unquoted case:
                int nextDelimiterIndex;
                if (_delimiterSet)
                {
                    // should optimize this:
                    nextDelimiterIndex = _curLine.IndexOfAny(_delimiter.ToCharArray(), curPos);
                }
                else
                {
                    nextDelimiterIndex = _curLine.IndexOf(_delimiter, curPos);
                }
                if (nextDelimiterIndex < 0)
                {
                    nextDelimiterIndex = _curLine.Length;
                }
                if (!_parseQuotes || _curLine.IndexOf(_quoteChar, curPos, nextDelimiterIndex - curPos) < 0)
                {
                    // simple case! no quote characters...
                    resStr = _curLine.Substring(curPos, nextDelimiterIndex - curPos);
                    curPos = nextDelimiterIndex;
                }
                else
                {
                    StringBuilder res = new StringBuilder();
                    for (bool inQuoted = false; true; curPos++)
                    {
                        // check for end of line:
                        if (curPos >= _curLine.Length)
                        {
                            if (!inQuoted)
                            {
                                // done with item!
                                break;
                            }
                            else
                            {
                                // uh oh... a quoted newline...
                                if (_file == null)
                                    break;
                                if (_file.Peek() == -1)
                                    break;
                                // read a new row...
                                // rowNumber++; -- don't count it as a new row.
                                _curLine = _file.ReadLine();
                                curPos = -1;
                                res.Append('\n');
                                continue;
                            }
                        }

                        // check for delimiters:
                        if (!inQuoted)
                        {
                            if (_delimiterSet)  // look for *any* of the characters
                            {
                                if (_curLine.Length - curPos >= 1 &&
                                    _delimiter.IndexOf(_curLine[curPos]) >= 0)
                                {
                                    // leave cursor on the delimiter
                                    break;
                                }
                            }
                            else
                            {
                                if (_curLine.Length - curPos >= _delimiter.Length &&
                                    string.CompareOrdinal(_curLine, curPos, _delimiter, 0, _delimiter.Length) == 0)
                                {
                                    // leave cursor on end of the delimiter:
                                    curPos += _delimiter.Length - 1;
                                    break;
                                }
                            }
                        }
                        // hard to say how to best handle quotes... Want to be lenient.
                        if (_parseQuotes &&
                            (_curLine.Length - curPos >= _quoteChar.Length &&
                            string.CompareOrdinal(_curLine, curPos, _quoteChar, 0, _quoteChar.Length) == 0))
                        {
                            // double quote characters mean a literal quote character
                            // - but only when in quoted mode!
                            if (inQuoted &&
                                _curLine.Length - curPos >= 2 * _quoteChar.Length &&
                                string.CompareOrdinal(_curLine, curPos + _quoteChar.Length, _quoteChar, 0, _quoteChar.Length) == 0)
                            {
                                res.Append(_quoteChar);
                                curPos += 2 * _quoteChar.Length - 1;
                            }
                            else
                            {
                                inQuoted = !inQuoted;
                                curPos += _quoteChar.Length - 1;
                            }
                            continue;
                        }
                        res.Append(_curLine[curPos]);
                    }
                    resStr = res.ToString();
                }
                if (_trimWhitespace)
                {
                    // not perfect! ***
                    if (resStr.Length > 0 && (resStr[0] == ' ' || resStr[resStr.Length - 1] == ' '))
                    {
                        resStr = resStr.Trim();
                    }
                }
            }
            // if we are collapsing repeated delimiters, we can do this by skipping blank elements...
            if (_collapseDelimiters && resStr.Length == 0)
            {
                return ReadItemFromLine(ref curPos);
            }
            //Console.Out.WriteLine("   Item:  " + resStr);
            return resStr;
        }

        /// <summary>
        /// Get the next field and advance the reader, filling with empty fields at the end of the row.
        /// </summary>
        /// <returns>the field at the next column</returns>
        public string ReadItemLinear()
        {
            if (!_initialized)
                Initialize();

            if (RowEnd())
                NextRow();
            return ReadItem();
        }

        /// <summary>
        /// Get the number of fields in the current row.
        /// </summary>
        /// <returns>the number of fields in the current row</returns>
        public int RowLength()
        {
            if (!_initialized)
                Initialize();

            if (_curRow == null)
            {
                if (_curRowArray == null)
                    return 0;
                return _curRowArray.Length;
            }
            return _curRow.Count;
        }

        // test if all entries in current row are blank
        private bool RowBlank()
        {
            if (!_initialized)
                Initialize();

            if (_curRow == null)
            {
                if (_curRowArray == null)
                    return true;
                for (int i = 0; i < _curRowArray.Length; i++)
                {
                    string field = _curRowArray[i];
                    if (field.Length != 0)
                    {
                        if (_trimWhitespace &&
                            (field[0] == ' ' || field[field.Length - 1] == ' '))
                        {
                            if (field.Trim().Length != 0)
                                return false;
                        }
                        else
                        {
                            return false;
                        }
                    }
                }
                return true;
            }
            for (int i = 0; i < _curRow.Count; i++)
            {
                string field = _curRow[i];
                if (field.Length != 0)
                {
                    if (_trimWhitespace &&
                        (field[0] == ' ' || field[field.Length - 1] == ' '))
                    {
                        if (field.Trim().Length != 0)
                            return false;
                    }
                    else
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        /// <summary>
        /// Read an entire row and advance the reader.
        /// </summary>
        /// <returns>The current row as an array of fields</returns>
        public string[] ReadRow()
        {
            return ReadRow(-1);
        }

        /// <summary>
        /// Read an entire row and advance the reader.
        /// </summary>
        /// <param name="len">The length of the row to read, truncating or filling with empty fields as needed</param>
        /// <returns>The current row as an array of fields</returns>
        public string[] ReadRow(int len)
        {
            // check if we need to setup the reading:
            if (!_initialized)
            {
                Initialize();
            }
            if (Eof())
                return null;
            if (_curRow == null && _curRowArray == null)
            {
                NextRow();
            }
            if (_curRow == null && _curRowArray == null)
                return null;

            if (!_parseQuotes && (_delimiterSet || _delimiter.Length == 1) && _curRow == null)
            {
                _readRowResult = _curRowArray;
                NextRow();
                return _readRowResult;
            }

            if (_curRow == null)
            {
                if (len < 0 || len == _curRowArray.Length)
                {
                    _readRowResult = _curRowArray;
                }
                else
                {
                    if (_readRowResult.Length != len)
                        _readRowResult = new string[len];
                    if (len < _curRowArray.Length)
                    {
#if ENABLE_BARTOK
						for (int i = 0; i < readRowResult.Length; i++)
						{
							readRowResult[i] = curRowArray[i];
						}
#else
                        Array.Copy(_curRowArray, _readRowResult, len);
#endif
                    }
                    else
                    {
                        _curRowArray.CopyTo(_readRowResult, 0);
                        for (int i = _curRowArray.Length; i < _readRowResult.Length; i++)
                        {
                            _readRowResult[i] = "";
                        }
                    }
                }
                NextRow();
                return _readRowResult;
            }
            // always make a new array, for now...
            _readRowResult = new string[len < 0 ? _curRow.Count : len];
            if (len < 0 || len == _curRow.Count)
            {
                if (_readRowResult.Length != _curRow.Count)
                    _readRowResult = new string[_curRow.Count];
                _curRow.CopyTo(_readRowResult, 0);
            }
            else
            {
                if (_readRowResult.Length != len)
                    _readRowResult = new string[len];
                if (len < _curRow.Count)
                {
                    for (int i = 0; i < _readRowResult.Length; i++)
                    {
                        _readRowResult[i] = _curRow[i];
                    }
                }
                else
                {
                    _curRow.CopyTo(_readRowResult, 0);
                    for (int i = _curRow.Count; i < _readRowResult.Length; i++)
                    {
                        _readRowResult[i] = "";
                    }
                }
            }
            NextRow();
            return _readRowResult;
        }
        string[] _readRowResult = new string[0];

        // only reads the *remaining* items...
        // also advances the row!
        private void ReadRowFromLine()
        {
            _curRow = null;
            _curRowArray = null;
            if (_curLine == null)
                return;
            if (_file == null)
                return;

            if (_parseQuotes || !(_delimiterSet || _delimiter.Length == 1))
            {
                _curRow = new StringCollection();
                int linePos = -1;
                while (linePos < _curLine.Length - 1)
                {
                    string field = ReadItemFromLine(ref linePos);
                    if (field != null)
                    {
                        _curRow.Add(field);
                    }
                }
            }
            else
            {
                // we just split, now...
                if (_delimiter.Length == 1)
                {
                    _curRowArray = _curLine.Split(_delimiter[0]);
                }
                else
                {
                    _curRowArray = _curLine.Split(_delimiter.ToCharArray());
                }
            }
            _curLine = null;
            _curCol = 0;
        }

        /// <summary>
        /// Advance to the next row.
        /// </summary>
        public void NextRow()
        {
            if (_file == null)
                return;
            if (!_initialized)
            {
                Initialize();
                return;
            }
            _curRow = null;
            _curRowArray = null;
            _curCol = 0;
            while (true)
            {
                _rowNumber++;
                _curLine = _file.ReadLine();
                if (_curLine == null)
                {
                    return;
                }
                if (_skipBlankLines &&
                    (_curLine.Length == 0 ||
                    (_curLine[0] == ' ' && _curLine.Trim().Length == 0)))
                {
                    //Console.Out.WriteLine("(Skipping row as blank line...)");
                    continue;
                }
                ReadRowFromLine();
                if (_skipBlankColumnsLines && RowBlank())
                {
                    //Console.Out.WriteLine("(Skipping row with blank columns...)");
                    _curRow = null;
                    _curRowArray = null;
                    continue;
                }
                break;
            }

            //ReadRowFromLine();
        }

        /// <summary>
        /// Close the table.
        /// </summary>
        public void Close()
        {
            if (_file == null)
                return;
            _file.Close();
            _curCol = -1;
            _curLine = null;
            _curRow = null;
            _curRowArray = null;
            _file = null;
        }

        /// <summary>
        /// Reset the reader to the beginning.
        /// </summary>
        /// <exception cref="InvalidOperationException">The reader is not based on a Stream.</exception>
        public void Reset()
        {
            if (_file == null)
                return;
            if (!(_file is StreamReader))
            {
                throw new InvalidOperationException("Cannot reset CsvReader not based on a Stream.");
            }
            ((StreamReader)_file).BaseStream.Seek(0, SeekOrigin.Begin);
            ((StreamReader)_file).DiscardBufferedData();
            _initialized = false;
            _rowNumber = 0;
            _curCol = -1;
            _curLine = null;
            _curRow = null;
            _curRowArray = null;
        }

        /// <summary>
        /// Allows for random access into the file. The user is responsible
        /// to make sure that position is at the beginning of a row.
        /// </summary>
        /// <param name="position">new position</param>
        /// <param name="origin">relative to what</param>
        /// <exception cref="InvalidOperationException">The reader is not based on a Stream.</exception>
        public void Seek(long position, SeekOrigin origin)
        {
            if (_file == null)
                return;
            if (!(_file is StreamReader))
            {
                throw new InvalidOperationException("Cannot seek reader not based on a Stream.");
            }

            ((StreamReader)_file).BaseStream.Seek(position, origin);
            if (position == 0)
            {
                _initialized = false;
            }
            _curCol = -1;
            _curLine = null;
            _curRow = null;
            _curRowArray = null;
        }

        /// <summary>
        /// Returns position of the cursor in the file
        /// </summary>
        /// <returns>byte offset of the current position</returns>
        /// <exception cref="InvalidOperationException">The reader is not based on a Stream.</exception>
        public long Position()
        {
            if (_file == null)
                return 0;
            if (!(_file is StreamReader))
            {
                throw new InvalidOperationException("Cannot tell position for reader not based on a Stream.");
            }
            return ((StreamReader)_file).BaseStream.Position;
        }

        #region IDisposable Members

        /// <summary>
        /// Dispose.
        /// </summary>
        public void Dispose()
        {
            Close();
        }

        #endregion
        /// <summary>
        /// Return the enumerator
        /// </summary>
        /// <returns>an enumerator for the rows in this table</returns>
        public IEnumerator GetEnumerator()
        {
            return new TableEnumerator(this);
        }
    }

    //// TODO: support reorder list
    //// TODO: support filtering (ignoring) columns
    /// <summary>
    /// Write CSV formatted data.
    /// </summary>
    public class CsvWriter : ITableWriter, IDisposable
    {
        // trim whitespace from each entry:
        private bool _trimWhitespace = true;
        // skip lines that are only whitespace:
        private bool _skipBlankLines = true;
        // use the quoteChar to determine quoted sections:
        private bool _parseQuotes = true;
        // always end in a newline:
        private bool _endInNewline = true;
        // transform tab, carriage return, and newline into space:
        private bool _normalizeWhitespace = false;
        // characters to use if normalizing whitespace:
        private char[] _whitespaceChars = new char[] { '\r', '\n', '\t' };

        // support for addressing by column headers:
        private string[] _headers = null;
        private string[] _headersNormalized = null;
        private bool _ignoreHeaderCase = true;
        private bool _initialized = false;
        private StringCollection _curRow = null;
        private bool _writeHeaders = true;

        private string _quoteChar = "\"";
        private string _delimiter = ",";
        private char[] _delimiterOrQuoteOrNewline = new char[] { ',', '"', '\r', '\n' };

        private bool _lineStart = true;
        private long _rowNumber;

        private TextWriter _file;

        /// <summary>
        /// Create a CsvWriter based on the TextWriter,
        /// </summary>
        /// <param name="tr">the TextWriter to write the table to</param>
        public CsvWriter(TextWriter tr)
        {
            _file = tr;
            //if (file == null)
            //	Console.Out.WriteLine("  Error: CsvReader could not open null file!");
            _rowNumber = 1;
        }

        /// <summary>
        /// Create a CsvWriter based on the specified file,
        /// </summary>
        /// <param name="fname">the name of the file to write the table to</param>
        public CsvWriter(string fname)
            : this(ZStreamWriter.Open(fname))
        {
        }

        /// <summary>
        /// Create a CsvWriter based on the Stream,
        /// </summary>
        /// <param name="fstream">the Stream to write the table to</param>
        public CsvWriter(Stream fstream)
            : this(new StreamWriter(fstream))
        {
        }

        /// <summary>
        /// Create a CsvWriter based on the specified file,
        /// </summary>
        /// <param name="fname">the name of the file to write the table to</param>
        /// <param name="encoding">the encoding to use</param>
        public CsvWriter(string fname, Encoding encoding)
            : this(new StreamWriter(fname, false, encoding))
        {
        }

        /// <summary>
        /// Create a CsvWriter based on the Stream,
        /// </summary>
        /// <param name="fstream">the Stream to write the table to</param>
        /// <param name="encoding">the encoding to use</param>
        public CsvWriter(Stream fstream, Encoding encoding)
            : this(new StreamWriter(fstream, encoding))
        {
        }

        /// <summary>
        /// Get or set whether to convert all whitespace into space characters.
        /// </summary>
        public bool NormalizeWhitespace
        {
            get { return _normalizeWhitespace; }
            set { _normalizeWhitespace = value; }
        }
        /// <summary>
        /// Get or set the characters to consider as whitespace.
        /// </summary>
        public char[] WhitespaceChars
        {
            get { return _whitespaceChars; }
            set
            {
                if (value != null)
                {
                    _whitespaceChars = value;
                }
                else
                {
                    _whitespaceChars = new char[] { '\r', '\n', '\t', ' ' };
                }
            }
        }

        /// <summary>
        /// Get or set whether to write the headers as the first row.
        /// </summary>
        public bool WriteHeaders
        {
            get { return _writeHeaders; }
            set { _writeHeaders = value; }
        }

        /// <summary>
        /// Gets or sets whether to trim whitespace from each field.
        /// </summary>
        public bool TrimWhitespace
        {
            get { return _trimWhitespace; }
            set { _trimWhitespace = value; }
        }

        /// <summary>
        /// Get or set whether to skip blank lines.
        /// </summary>
        public bool SkipBlankLines
        {
            get { return _skipBlankLines; }
            set { _skipBlankLines = value; }
        }

        /// <summary>
        /// Get or set whether to interpet quote characters when parsing.
        /// </summary>
        public bool ParseQuotes
        {
            get { return _parseQuotes; }
            set { _parseQuotes = value; }
        }

        /// <summary>
        /// Get or set the string to use as a quote symbol.
        /// </summary>
        public string QuoteChar
        {
            get { return _quoteChar; }
            set
            {
                _quoteChar = value;
                FixupDelimiterOrQuoteOrNewline();
            }
        }

        /// <summary>
        /// Get or set the string to use to delimit columns.
        /// </summary>
        public string Delimiter
        {
            get { return _delimiter; }
            set
            {
                _delimiter = value;
                FixupDelimiterOrQuoteOrNewline();
            }
        }
        private void FixupDelimiterOrQuoteOrNewline()
        {
            string dqn = (_delimiter == null ? "" : _delimiter) + (_quoteChar == null ? "" : _quoteChar) + "\r\n";
            _delimiterOrQuoteOrNewline = dqn.ToCharArray();
        }

        /// <summary>
        /// Get or set whether to end the file in a newline.
        /// </summary>
        public bool EndInNewline
        {
            get { return _endInNewline; }
            set { _endInNewline = value; }
        }

        /// <summary>
        /// Gets or sets whether to ignore case when matching header names.
        /// </summary>
        public bool IgnoreHeaderCase
        {
            get { return _ignoreHeaderCase; }
            set
            {
                if (_ignoreHeaderCase != value)
                {
                    _ignoreHeaderCase = value;
                    FixupNormalizedHeaders();
                }
            }
        }
        /// <summary>
        /// Gets or sets the header names.
        /// </summary>
        public string[] Headers
        {
            get
            {
                return _headers;
            }
            set
            {
                _headers = value;
                FixupNormalizedHeaders();
            }
        }
        private void FixupNormalizedHeaders()
        {
            if (_headers == null)
            {
                _headersNormalized = null;
                return;
            }
            _headersNormalized = new string[_headers.Length];
            for (int i = 0; i < _headers.Length; i++)
            {
                string header = _headers[i];
                if (header == null)
                    continue;
                header = header.Trim();
                if (IgnoreHeaderCase)
                    header = header.ToLower();
                _headersNormalized[i] = header;
            }
        }

        /// <summary>
        /// Add a new header to the header list.
        /// </summary>
        /// <param name="header">the header to add</param>
        public void AddHeader(string header)
        {
            if (_headers == null)
                _headers = new string[0];
            string[] newHeaders = new string[_headers.Length + 1];
            _headers.CopyTo(newHeaders, 0);
            newHeaders[newHeaders.Length - 1] = header;
            Headers = newHeaders;
        }

        /// <summary>
        /// Check for end of file.
        /// </summary>
        /// <returns>true if at end of file, false otherwise</returns>
        public bool Eof()
        {
            return (_file == null);
        }

        private void Initialize()
        {
            if (!_initialized)
            {
                _initialized = true;
                // check if headers were needed and not written:
                if (_writeHeaders && _headers != null)
                {
                    WriteRow(_headers);
                }
            }
        }

        private string Quotify(string item)
        {
            if (_trimWhitespace)
            {
                if (item.Length != 0 &&
                    (item[0] == ' ' || item[item.Length - 1] == ' '))
                {
                    item = item.Trim();
                }
            }
            if (!_parseQuotes)
            {
                return item;
            }

            if (item.IndexOfAny(_delimiterOrQuoteOrNewline) < 0)
            {
                // no delimiters, newlines, or quotes in string
                return item;
            }
            StringBuilder sb = new StringBuilder(item);
            sb.Replace(_quoteChar, _quoteChar + _quoteChar);
            sb.Insert(0, _quoteChar);
            sb.Append(_quoteChar);
            return sb.ToString();
        }

        /// <summary>
        /// Write the next field and advance the writer.
        /// </summary>
        /// <param name="item">the field to write</param>
        public void WriteItem(string item)
        {
            Initialize();
            WriteStoredRow();
            if (Eof())
                return;

            if (_lineStart)
            {
                _lineStart = false;
            }
            else
            {
                _file.Write(_delimiter);
            }

            if (_normalizeWhitespace)
            {
                item = NormalizeWS(item);
            }
            item = Quotify(item);
            _file.Write(item);
        }

        private string NormalizeWS(string orig)
        {
            if (orig.IndexOfAny(_whitespaceChars) < 0)
                return orig;
            orig = orig.Replace("\r\n", " ");
            orig = orig.Replace('\r', ' ');
            orig = orig.Replace('\n', ' ');
            orig = orig.Replace('\t', ' ');
            return orig;
        }

        /// <summary>
        /// Write an entire row and advance the writer.
        /// </summary>
        /// <param name="items">The row to write as an array of fields</param>
        public void WriteRow(string[] items)
        {
            WriteRow(items, -1);
        }

        /// <summary>
        /// Write an entire row and advance the writer.
        /// </summary>
        /// <param name="len">The length of the row to write, truncating or filling with empty fields as needed</param>
        /// <param name="items">The row to write as an array of fields</param>
        public void WriteRow(string[] items, int len)
        {
            Initialize();
            WriteStoredRow();
            if (Eof())
                return;

            if (!_parseQuotes)
            {
                if (_normalizeWhitespace)
                {
                    // this modifies the input parameter!
                    for (int i = 0; i < items.Length; i++)
                    {
                        items[i] = NormalizeWS(items[i]);
                    }
                }

                if (_skipBlankLines && _lineStart && items.Length == 0)
                    return;
                //				if (len < 0 || len == items.Length)
                //				{
                //					file.WriteLine(string.Join(delimiter, items));
                //				}
                //				else
                //				{
                //					if (items.Length < len)
                //					{
                //						file.Write(string.Join(delimiter, items));
                //						// pad with empty strings:
                //						for (int i = items.Length; i < len; i++)
                //						{
                //							file.Write(delimiter);
                //						}
                //						file.WriteLine();
                //					}
                //					else
                //					{
                //						file.WriteLine(string.Join(delimiter, items, 0, len));
                //					}
                //				}
                if (items.Length == 0)
                {
                    if (len > 0)
                    {
                        for (int i = 1; i < len; i++)
                        {
                            _file.Write(_delimiter);
                        }
                    }
                }
                else
                {
                    _file.Write(items[0]);
                    if (len < 0 || len >= items.Length)
                    {
                        for (int i = 1; i < items.Length; i++)
                        {
                            _file.Write(_delimiter);
                            if (items[i].Length != 0)
                                _file.Write(items[i]);
                        }
                        for (int i = items.Length; i < len; i++)
                        {
                            _file.Write(_delimiter);
                        }
                    }
                    else
                    {
                        for (int i = 1; i < len; i++)
                        {
                            _file.Write(_delimiter);
                            if (items[i].Length != 0)
                                _file.Write(items[i]);
                        }
                    }
                }
                _file.WriteLine();

                _lineStart = true;
                _rowNumber++;
                return;
            }

            for (int i = 0; i < items.Length && (len < 0 || i < len); i++)
            {
                WriteItem(items[i]);
            }
            // pad with empty strings:
            if (len > 0)
            {
                for (int i = items.Length; i < len; i++)
                {
                    WriteItem("");
                }
            }
            NextRow();
        }

        /// <summary>
        /// Write an entire row and advance the writer.
        /// </summary>
        /// <param name="items">The row to write as a collection of fields</param>
        public void WriteRow(StringCollection items)
        {
            WriteRow(items, -1);
        }

        /// <summary>
        /// Write an entire row and advance the writer.
        /// </summary>
        /// <param name="len">The length of the row to write, truncating or filling with empty fields as needed</param>
        /// <param name="items">The row to write as a collection of fields</param>
        public void WriteRow(StringCollection items, int len)
        {
            Initialize();
            WriteStoredRow();
            if (Eof())
                return;

            if (!_parseQuotes)
            {
                //				string[] itemsA = new string[items.Count];
                //				items.CopyTo(itemsA, 0);

                if (_normalizeWhitespace)
                {
                    //					for (int i = 0; i < itemsA.Length; i++)
                    //					{
                    //						itemsA[i] = NormalizeWS(itemsA[i]);
                    //					}
                    for (int i = 0; i < items.Count; i++)
                    {
                        items[i] = NormalizeWS(items[i]);
                    }
                }

                //				if (skipBlankLines && lineStart && itemsA.Length == 0)  return;
                if (_skipBlankLines && _lineStart && items.Count == 0)
                    return;
                //				if (len < 0 || len == itemsA.Length)
                //				{
                //					file.WriteLine(string.Join(delimiter, itemsA));
                //				}
                //				else
                //				{
                //					if (itemsA.Length < len)
                //					{
                //						file.Write(string.Join(delimiter, itemsA));
                //						// pad with empty strings:
                //						for (int i = itemsA.Length; i < len; i++)
                //						{
                //							file.Write(delimiter);
                //						}
                //						file.WriteLine();
                //					}
                //					else
                //					{
                //						file.WriteLine(string.Join(delimiter, itemsA, 0, len));
                //					}
                //				}
                if (items.Count == 0)
                {
                    if (len > 0)
                    {
                        for (int i = 1; i < len; i++)
                        {
                            _file.Write(_delimiter);
                        }
                    }
                }
                else
                {
                    _file.Write(items[0]);
                    if (len < 0 || len >= items.Count)
                    {
                        for (int i = 1; i < items.Count; i++)
                        {
                            _file.Write(_delimiter);
                            if (items[i].Length != 0)
                                _file.Write(items[i]);
                        }
                        for (int i = items.Count; i < len; i++)
                        {
                            _file.Write(_delimiter);
                        }
                    }
                    else
                    {
                        for (int i = 1; i < len; i++)
                        {
                            _file.Write(_delimiter);
                            if (items[i].Length != 0)
                                _file.Write(items[i]);
                        }
                    }
                }
                _file.WriteLine();

                _lineStart = true;
                _rowNumber++;
                return;
            }

            for (int i = 0; i < items.Count && (len < 0 || i < len); i++)
            {
                WriteItem(items[i]);
            }
            // pad with empty strings:
            if (len > 0)
            {
                for (int i = items.Count; i < len; i++)
                {
                    WriteItem("");
                }
            }
            NextRow();
        }

        /// <summary>
        /// Set the field at the column index.
        /// </summary>
        public string this[int index]
        {
            set
            {
                if (index < 0)
                    return;
                if (_curRow == null)
                    _curRow = new StringCollection();
                if (_curRow.Count <= index)  // extend it...
                {
                    for (int i = _curRow.Count; i <= index; i++)
                    {
                        _curRow.Add("");
                    }
                }
                _curRow[index] = value;
            }
        }

        /// <summary>
        /// Set the field at the column with the given header.
        /// </summary>
        public string this[string header]
        {
            set
            {
                if (header == null || _headers == null)
                    return;
                header = header.Trim();
                if (IgnoreHeaderCase)
                    header = header.ToLower();
                for (int i = 0; i < _headersNormalized.Length; i++)
                {
                    if (header == _headersNormalized[i])
                    {
                        this[i] = value;
                        return;
                    }
                }
                // do nothing!
                return;
            }
        }

        private void WriteStoredRow()
        {
            if (_curRow == null)
                return;
            WriteRow(_curRow);
            _curRow = null;
        }

        /// <summary>
        /// Advance to the next row.
        /// </summary>
        public void NextRow()
        {
            if (Eof())
                return;
            if (_curRow != null)
            {
                WriteStoredRow();
                return;
            }
            if (_lineStart && _skipBlankLines)
                return;
            _file.WriteLine();
            _lineStart = true;
            _rowNumber++;
        }

        /// <summary>
        /// Get the number of the current row.
        /// </summary>
        public long RowNumber
        {
            get { return _rowNumber; }
        }

        /// <summary>
        /// Close the table.
        /// </summary>
        public void Close()
        {
            if (Eof())
                return;
            if (_curRow != null)
            {
                WriteStoredRow();
                return;
            }
            if (!_lineStart && _endInNewline)
                NextRow();
            _file.Flush();
            _file.Close();
            _lineStart = true;
            _file = null;
        }
        #region IDisposable Members

        /// <summary>
        /// Dispose.
        /// </summary>
        public void Dispose()
        {
            Close();
        }

        #endregion
    }

#if !ENABLE_BARTOK

#if !DISABLE_XML
    //////////////////////////////////////////////
    ////// XML support
    //////////////////////////////////////////////

    //// TODO: Decide how to handle attributes vs. child elements
    //// TODO: Allow for categories (pushing and popping levels)
    //// TODO: Allow override of given element names by header order?

    /// <summary>
    /// Read XML formatted data.
    /// </summary>
    public class XmlTableReader : ITableReader
    {
        private bool _trimWhitespace = true;
        private bool _fillBlankColumns = true;
        private bool _ignoreHeaderCase = true;
        private bool _addUnknownHeaders = true;
        private bool _headerOrdered = true;

        private XmlTextReader _file;
        private Hashtable _currentRow;
        private ArrayList _currentRowSequence;
        private string _currentName;
        private int _currentCol;
        //private bool          initialized = false;
        private string _tableName;
        private string[] _headers;
        private bool _initialized = false;

        private int _level = 0;

        /// <summary>
        /// Create a new XmlTableReader
        /// </summary>
        /// <param name="tr">the source to base the XmlTableReader on</param>
        public XmlTableReader(XmlTextReader tr)
        {
            tr.DtdProcessing = DtdProcessing.Prohibit;
            _headers = new string[0];
            _file = tr;
            if (_file != null)
            {
                _file.WhitespaceHandling = WhitespaceHandling.None;
                _file.MoveToContent();
                // read in the "Table" element
                _tableName = "";
                if (_file.IsStartElement())
                {
                    _tableName = _file.Name;
                    _file.Read();
                }

                // NextRow();  // - don't do this! Might initialize too soon.
            }
        }

        /// <summary>
        /// Create a new XmlTableReader
        /// </summary>
        /// <param name="fname">the name of the file to base the XmlTableReader on</param>
        public XmlTableReader(string fname)
            : this(new XmlTextReader(ZStreamReader.Open(fname)) { DtdProcessing = DtdProcessing.Prohibit })
        {
        }

        /// <summary>
        /// Create a new XmlTableReader
        /// </summary>
        /// <param name="tr">the source to base the XmlTableReader on</param>
        public XmlTableReader(Stream tr)
            : this(new XmlTextReader(tr) { DtdProcessing = DtdProcessing.Prohibit })
        {
        }

        /// <summary>
        /// Create a new XmlTableReader
        /// </summary>
        /// <param name="tr">the source to base the XmlTableReader on</param>
        public XmlTableReader(TextReader tr)
            : this(new XmlTextReader(tr) { DtdProcessing = DtdProcessing.Prohibit })
        {
        }

        /// <summary>
        /// Get the current in the hierarchy.
        /// </summary>
        public int Level
        {
            get { return _level; }
        }

        /// <summary>
        /// Gets or sets the header names.
        /// </summary>
        public string[] Headers
        {
            get { return _headers; }
            set { _headers = value; }
        }

        /// <summary>
        /// Get or set whether to add new headers to the header list as they are encountered.
        /// </summary>
        public bool AddUnknownHeaders
        {
            get { return _addUnknownHeaders; }
            set { _addUnknownHeaders = value; }
        }

        /// <summary>
        /// Get or set whether to sort the headers.
        /// </summary>
        public bool HeaderOrdered
        {
            get { return _headerOrdered; }
            set { _headerOrdered = value; }
        }

        /// <summary>
        /// Get the name of the table.
        /// </summary>
        public string TableName
        {
            get { return _tableName; }
        }

        /// <summary>
        /// Check for end of file.
        /// </summary>
        /// <returns>true if at end of file, false otherwise</returns>
        public bool Eof()
        {
            if (!_initialized)
                Initialize();
            return _currentRow == null && (_file.ReadState == ReadState.Closed || _file.EOF);
        }

        /// <summary>
        /// Check for end of row.
        /// </summary>
        /// <returns>true if at end of row, false otherwise</returns>
        public bool RowEnd()
        {
            if (!_initialized)
                Initialize();
            return Eof() || _currentRowSequence == null ||
                _currentCol >= _currentRowSequence.Count;
        }

        /// <summary>
        /// Get the next field and advance the reader.
        /// </summary>
        /// <returns>the field at the next column</returns>
        public string ReadItem()
        {
            if (!_initialized)
                Initialize();
            if (Eof())
                return null;
            if (RowEnd())
            {
                // should we fill even this??
                if (FillBlankColumns)
                    return "";
                return null;
            }
            string res = (string)_currentRowSequence[_currentCol];
            _currentCol++;
            return res;
        }

        /// <summary>
        /// Get the next field and advance the reader, filling with empty fields at the end of the row.
        /// </summary>
        /// <returns>the field at the next column</returns>
        public string ReadItemLinear()
        {
            if (!_initialized)
                Initialize();
            if (Eof())
                return null;
            if (RowEnd())
                NextRow();
            return ReadItem();
        }

        /// <summary>
        /// Get the number of fields in the current row.
        /// </summary>
        /// <returns>the number of fields in the current row</returns>
        public int RowLength()
        {
            if (!_initialized)
                Initialize();
            if (_currentRowSequence == null)
                return 0;
            return _currentRowSequence.Count;
        }

        /// <summary>
        /// Read an entire row and advance the reader.
        /// </summary>
        /// <returns>The current row as an array of fields</returns>
        public string[] ReadRow()
        {
            if (!_initialized)
                Initialize();
            if (_currentRowSequence == null)
                return null;
            string[] res = (string[])_currentRowSequence.ToArray(typeof(string));
            NextRow();
            return res;
        }

        /// <summary>
        /// Read an entire row and advance the reader.
        /// </summary>
        /// <param name="len">The length of the row to read, truncating or filling with empty fields as needed</param>
        /// <returns>The current row as an array of fields</returns>
        public string[] ReadRow(int len)
        {
            if (!_initialized)
                Initialize();
            if (len < 0)
                return ReadRow();
            if (_currentRowSequence == null)
                return null;
            string[] res = new string[len];
            for (int i = 0; i < res.Length; i++)
            {
                if (i < _currentRowSequence.Count)
                {
                    res[i] = (string)_currentRowSequence[i];
                }
                else
                {
                    res[i] = "";
                }
            }
            NextRow();
            return res;
        }

        private void Initialize()
        {
            if (_initialized)
                return;
            _initialized = true;
            NextRow();
            // can't read and store the column header names:
        }

        /// <summary>
        /// Advance to the next row.
        /// </summary>
        public void NextRow()
        {
            if (!_initialized)
            {
                Initialize();
                return;
            }
            _currentRow = null;
            _currentRowSequence = null;
            _currentName = null;
            _currentCol = -1;
            if (Eof())
            {
                return;
            }

            if (!_file.IsStartElement())
            {
                // what to do here? It's wrong.
                _file.Read();
                NextRow();
                return;
            }

            _currentName = _file.Name;
            _currentRow = new Hashtable();
            // assumes headers not null
            _currentRowSequence = new ArrayList(_headers.Length);
            _currentCol = 0;
            if (!_file.IsEmptyElement)
            {
                string givenHeader, name, val;

                if (_file.HasAttributes)
                {
                    for (int a = 0; a < _file.AttributeCount; a++)
                    {
                        _file.MoveToAttribute(a);
                        givenHeader = _file.Name;
                        name = givenHeader.Trim();
                        if (IgnoreHeaderCase)
                            name = name.ToLower();
                        val = _file.Value;
                        if (TrimWhitespace)
                            val = val.Trim();

                        int index = -1;
                        for (int i = 0; i < _headers.Length; i++)
                        {
                            string match = _headers[i].Trim();
                            if (IgnoreHeaderCase)
                                match = match.ToLower();
                            if (name == match)
                            {
                                index = i;
                                break;
                            }
                        }
                        if (index < 0)
                        {
                            if (_addUnknownHeaders)
                            {
                                string[] oldHeaders = _headers;
                                _headers = new string[oldHeaders.Length + 1];
                                oldHeaders.CopyTo(_headers, 0);
                                index = _headers.Length - 1;
                                _headers[index] = givenHeader;
                            }
                        }
                        //currentRow.Add(name, val);
                        _currentRow[name] = val;
                        if (!_headerOrdered)
                        {
                            // skip adding to the sequence for attributes!
                            //currentRowSequence.Add(val);
                        }
                        else
                        {
                            if (val.Length != 0 && index >= 0 && index < _headers.Length)
                            {
                                for (int b = _currentRowSequence.Count; b <= index; b++)
                                {
                                    _currentRowSequence.Add("");
                                }
                                _currentRowSequence[index] = val;
                            }
                        }
                    }
                    _file.MoveToElement();  // Moves the reader back to the element node.
                }

                _file.Read();  // pass start element
                while (!(_file.NodeType == XmlNodeType.EndElement))
                {
                    if (_file.IsStartElement())
                    {
                        givenHeader = _file.Name;
                        name = givenHeader.Trim();
                        if (IgnoreHeaderCase)
                            name = name.ToLower();
                        val = "";
                        if (!_file.IsEmptyElement)
                        {
                            _file.Read();
                            val = _file.ReadString();
                            if (TrimWhitespace)
                                val = val.Trim();
                        }

                        int index = -1;
                        for (int i = 0; i < _headers.Length; i++)
                        {
                            string match = _headers[i].Trim();
                            if (IgnoreHeaderCase)
                                match = match.ToLower();
                            if (name == match)
                            {
                                index = i;
                                break;
                            }
                        }
                        if (index < 0)
                        {
                            if (_addUnknownHeaders)
                            {
                                string[] oldHeaders = _headers;
                                _headers = new string[oldHeaders.Length + 1];
                                oldHeaders.CopyTo(_headers, 0);
                                index = _headers.Length - 1;
                                _headers[index] = givenHeader;
                            }
                        }
                        //currentRow.Add(name, val);
                        _currentRow[name] = val;
                        if (!_headerOrdered)
                        {
                            // skip adding to the sequence for attributes!
                            //currentRowSequence.Add(val);
                        }
                        else
                        {
                            if (val.Length != 0 && index >= 0 && index < _headers.Length)
                            {
                                for (int b = _currentRowSequence.Count; b <= index; b++)
                                {
                                    _currentRowSequence.Add("");
                                }
                                _currentRowSequence[index] = val;
                            }
                        }
                        _file.Read();  // skip empty element OR end element...
                    }
                    else
                    {
                        // text not within a child element??
                        val = _file.ReadString();
                        if (TrimWhitespace)
                            val = val.Trim();
                        //currentRow.Add("", val);
                        _currentRow[""] = val;
                    }
                }
            }
            _file.Read();
            // try to skip the document closing tag? (XmlReader is disgusting, really)
            while (_file.NodeType == XmlNodeType.EndElement)
            {
                _file.Read();
                _level--;
            }
            if (_level < 0)
                _level = 0;
        }

        /// <summary>
        /// Get the field at the column index.
        /// </summary>
        public string this[int index]
        {
            get
            {
                if (!_initialized)
                    Initialize();
                if (_currentRowSequence == null || index < 0 ||
                    index >= _currentRowSequence.Count)
                {
                    if (FillBlankColumns)
                        return "";
                    return null;
                }
                return (string)_currentRowSequence[index];
            }
        }

        /// <summary>
        /// Get the field at the column with the given header.
        /// </summary>
        public string this[string header]
        {
            get
            {
                if (!_initialized)
                    Initialize();
                header = header.Trim();
                if (IgnoreHeaderCase)
                    header = header.ToLower();
                string res = (string)_currentRow[header];
                if (res == null && FillBlankColumns)
                    res = "";
                return res;
            }
        }

        /// <summary>
        /// Close the table.
        /// </summary>
        public void Close()
        {
            _file.Close();
        }

        /// <summary>
        /// Reset the reader to the beginning.
        /// </summary>
        /// <exception cref="InvalidOperationException">Always thrown, currently.</exception>
        public void Reset()
        {
            throw new InvalidOperationException("Cannot reset XmlTableReader.");
        }

        /// <summary>
        /// Gets or sets whether to trim whitespace from each field.
        /// </summary>
        public bool TrimWhitespace
        {
            get
            {
                return _trimWhitespace;
            }
            set
            {
                _trimWhitespace = value;
            }
        }

        /// <summary>
        /// Gets or sets whether to return "", not null, when the end of a row is reached,
        /// until the row is advanced:
        /// </summary>
        public bool FillBlankColumns
        {
            get
            {
                return _fillBlankColumns;
            }
            set
            {
                _fillBlankColumns = value;
            }
        }

        /// <summary>
        /// Gets or sets whether to ignore case when matching header names.
        /// </summary>
        public bool IgnoreHeaderCase
        {
            get
            {
                return _ignoreHeaderCase;
            }
            set
            {
                _ignoreHeaderCase = value;
            }
        }

        /// <summary>
        /// Get a row enumerator.
        /// </summary>
        /// <returns>and enumerator for the row in this table</returns>
        public IEnumerator GetEnumerator()
        {
            return new TableEnumerator(this);
        }
    }

    /// <summary>
    /// Write XML formatted data.
    /// </summary>
    public class XmlTableWriter : ITableWriter
    {
        private bool _trimWhitespace = true;
        private bool _ignoreHeaderCase = true;
        private bool _addUnknownHeaders = true;
        private bool _skipEmptyElements = true;

        private XmlTextWriter _file;
        //private Hashtable     currentRow;
        private ArrayList _currentRowSequence;
        private string _tableName = "Table";
        private string _currentName = "Item";
        private int _currentCol;
        private bool _initialized;
        private string[] _headers;
        private bool _fieldsAsAttributes;
        private bool _elementEndPending;
        private int _openElementCount;

        /// <summary>
        /// Create a new XmlTableWriter.
        /// </summary>
        /// <param name="tr">the destination to write the table to</param>
        public XmlTableWriter(XmlTextWriter tr)
        {
            _file = tr;
            _currentCol = 0;
            if (_file != null)
            {
                _file.Formatting = Formatting.Indented;
                _file.Indentation = 4;
                //Write the XML delcaration
                _file.WriteStartDocument();
            }
        }

        /// <summary>
        /// Create a new XmlTableWriter.
        /// </summary>
        /// <param name="fname">the filename of the destination to write the table to</param>
        public XmlTableWriter(string fname)
            : this(new XmlTextWriter(ZStreamWriter.Open(fname)))
        {
        }

        /// <summary>
        /// Create a new XmlTableWriter.
        /// </summary>
        /// <param name="tr">the destination to write the table to</param>
        public XmlTableWriter(Stream tr)
            : this(new XmlTextWriter(tr, null))
        {
        }

        /// <summary>
        /// Create a new XmlTableWriter.
        /// </summary>
        /// <param name="tr">the destination to write the table to</param>
        public XmlTableWriter(TextWriter tr)
            : this(new XmlTextWriter(tr))
        {
        }

        /// <summary>
        /// Gets or sets the header names.
        /// </summary>
        public string[] Headers
        {
            get
            {
                return _headers;
            }
            set
            {
                _headers = value;
                if (_headers != null)
                {
                    for (int i = 0; i < _headers.Length; i++)
                    {
                        if (_headers[i] == null || _headers[i].Length == 0)
                        {
                            _headers[i] = "C" + i;
                        }
                        else
                        {
                            _headers[i] = _headers[i].Replace(' ', '_');
                            _headers[i] = XmlConvert.EncodeName(_headers[i]);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Get or set the name to use for each item element,
        /// </summary>
        public string ItemName
        {
            get
            {
                return _currentName;
            }
            set
            {
                _currentName = value;
            }
        }

        /// <summary>
        /// Get or set the name to use for the table element.
        /// </summary>
        public string TableName
        {
            get
            {
                return _tableName;
            }
            set
            {
                _tableName = value;
            }
        }

        /// <summary>
        /// Get or set whether to represent the fields as attributes, instead of children.
        /// </summary>
        public bool FieldsAsAttributes
        {
            get { return _fieldsAsAttributes; }
            set { _fieldsAsAttributes = value; }
        }

        /// <summary>
        /// Get or set whether to skip all empty elements.
        /// </summary>
        public bool SkipEmptyElements
        {
            get { return _skipEmptyElements; }
            set { _skipEmptyElements = value; }
        }

        /// <summary>
        /// Check for end of file.
        /// </summary>
        /// <returns>true if at end of file, false otherwise</returns>
        public bool Eof()
        {
            return _file.WriteState == WriteState.Closed;
        }

        /// <summary>
        /// Write the next field and advance the writer.
        /// </summary>
        /// <param name="item">the field to write</param>
        public void WriteItem(string item)
        {
            if (Eof())
                return;
            this[_currentCol] = item;
            _currentCol++;
        }

        /// <summary>
        /// Write an entire row and advance the writer.
        /// </summary>
        /// <param name="items">The row to write as an array of fields</param>
        public void WriteRow(string[] items)
        {
            WriteRow(items, -1);
        }

        /// <summary>
        /// Write an entire row and advance the writer.
        /// </summary>
        /// <param name="len">The length of the row to write, truncating or filling with empty fields as needed</param>
        /// <param name="items">The row to write as an array of fields</param>
        public void WriteRow(string[] items, int len)
        {
            // only handles sequence - not named items!!! ***
            if (Eof())
                return;
            if (items == null)
                return;
            if (!_initialized)
            {
                _initialized = true;
                _file.WriteStartElement(TableName);
            }

            if (_elementEndPending)
            {
                if (_openElementCount > 0)
                {
                    if (TrimWhitespace)
                    {
                        _file.WriteEndElement();
                    }
                    else
                    {
                        _file.WriteFullEndElement();
                    }
                    _openElementCount--;
                }
                _elementEndPending = false;
            }

            if (len < 0)
                len = items.Length;
            _file.WriteStartElement(ItemName);
            for (int i = 0; i < len; i++)
            {
                string name;
                if (_headers != null && i < _headers.Length)
                {
                    name = _headers[i];
                }
                else
                {
                    name = "C" + i;
                }
                if (FieldsAsAttributes)
                {
                    string val = (i < items.Length) ? (string)items[i] : "";
                    if (val == null)
                        val = "";
                    if (TrimWhitespace)
                        val = val.Trim();
                    if (SkipEmptyElements && val.Length == 0)
                    {
                        // just skip it...
                    }
                    else
                    {
                        _file.WriteAttributeString(name, val);
                    }
                }
                else
                {
                    string val = (i < items.Length) ? (string)items[i] : "";
                    if (val == null)
                        val = "";
                    if (TrimWhitespace)
                        val = val.Trim();
                    ////file.WriteElementString(name, val);
                    if (SkipEmptyElements && val.Length == 0)
                    {
                        // just skip it...
                    }
                    else
                    {
                        _file.WriteStartElement(name);
                        if (val.Length == 0)
                        {
                            _file.WriteEndElement();
                        }
                        else
                        {
                            _file.WriteString(val);
                            _file.WriteFullEndElement();
                        }
                    }
                }
            }

            // delay writing the end element:
            //file.WriteEndElement();
            _elementEndPending = true;
            _openElementCount++;
        }

        /// <summary>
        /// Advance to the next row.
        /// </summary>
        public void NextRow()
        {
            if (_currentRowSequence != null)
            {
                // only handles sequence - not named items!!! ***
                WriteRow((string[])_currentRowSequence.ToArray(typeof(string)));
                _currentRowSequence = null;
            }

            //currentRow = null;
            _currentRowSequence = null;
            _currentCol = 0;
        }

        /// <summary>
        /// Set the field at the column index.
        /// </summary>
        public string this[int index]
        {
            // checking headers can be done here or at write time. Not much difference.
            set
            {
                if (index < 0)
                    return;
                if (_currentRowSequence == null)
                    _currentRowSequence = new ArrayList();
                if (_currentRowSequence.Count <= index)  // extend it...
                {
                    for (int i = _currentRowSequence.Count; i <= index; i++)
                    {
                        _currentRowSequence.Add("");
                    }
                }
                _currentRowSequence[index] = value;
            }
        }

        /// <summary>
        /// Set the field at the column with the given header.
        /// </summary>
        public string this[string header]
        {
            // checking headers can be done here or at write time. Not much difference.
            set
            {
                if (header == null)
                    return;
                //if (currentRow == null)  currentRow = new Hashtable();
                string givenHeader = header;
                header = header.Trim();
                if (IgnoreHeaderCase)
                    header = header.ToLower();
                // don't use hash table:
                //currentRow[header] = value;
                // find in headers:
                for (int i = 0; i < _headers.Length; i++)
                {
                    string match = _headers[i].Trim();
                    if (IgnoreHeaderCase)
                        match = match.ToLower();
                    if (header == match)
                    {
                        this[i] = value;
                        return;
                    }
                }
                // add new header if required:
                if (!AddUnknownHeaders)
                    return;
                string[] oldHeaders = _headers;
                _headers = new string[oldHeaders.Length + 1];
                oldHeaders.CopyTo(_headers, 0);
                _headers[_headers.Length - 1] = givenHeader;
                this[_headers.Length - 1] = value;
            }
        }

        /// <summary>
        /// Close the table.
        /// </summary>
        public void Close()
        {
            if (_file == null || Eof())
                return;
            // we don't really want this NextRow unless needed... ***
            NextRow();

            for (; _openElementCount > 0; _openElementCount--)
            {
                if (TrimWhitespace)
                {
                    _file.WriteEndElement();
                }
                else
                {
                    _file.WriteFullEndElement();
                }
            }
            _elementEndPending = false;

            _file.WriteFullEndElement();
            _file.WriteEndDocument();
            _file.Flush();
            _file.Close();
        }

        /// <summary>
        /// Increase the hierarchy depth.
        /// </summary>
        public void LevelIn()
        {
            _elementEndPending = false;
        }

        /// <summary>
        /// Decrease the hierarchy depth.
        /// </summary>
        public void LevelOut()
        {
            if (_openElementCount > 1 || (!_elementEndPending && _openElementCount > 0))
            {
                if (TrimWhitespace)
                {
                    _file.WriteEndElement();
                }
                else
                {
                    _file.WriteFullEndElement();
                }

                _openElementCount--;
            }
        }

        /// <summary>
        /// Gets or sets whether to trim whitespace from each field.
        /// </summary>
        public bool TrimWhitespace
        {
            get { return _trimWhitespace; }
            set { _trimWhitespace = value; }
        }

        /// <summary>
        /// Get or set whether to add new headers to the header list as they are encountered.
        /// </summary>
        public bool AddUnknownHeaders
        {
            get { return _addUnknownHeaders; }
            set { _addUnknownHeaders = value; }
        }

        /// <summary>
        /// Gets or sets whether to ignore case when matching header names.
        /// </summary>
        public bool IgnoreHeaderCase
        {
            get { return _ignoreHeaderCase; }
            set { _ignoreHeaderCase = value; }
        }
    }

#endif

#endif

#if ALLOW_DB
    //////////////////////////////////////////////
    ////// Excel support
    //////////////////////////////////////////////

    //// TODO: ODBC driver support?
    ////       Bare CsvReader support / merging?
    ////       Decent exception handling
    ////       Add in enumerator

    /// <summary>
    /// Read spreadsheet data in various formats.
    /// </summary>
    public class SpreadsheetReader
    {
        private string filename;
        private bool xlsFormat;
        private OleDbConnection oleConn;
        private OleDbDataReader oleReader;
        private bool isEof;
        private string[] cols;

        // trim whitespace from each entry:
        private bool trimWhitespace = true;
        // skip lines that are empty:
        private bool skipBlankLines = true;
        // skip lines that have fields but only whitespace in them:
        private bool skipBlankColumnsLines = true;
        private long rowNumber = 2;  // gives 1-base row numbers...

        /// <summary>
        /// Gets or sets whether to trim whitespace from each field.
        /// </summary>
        public bool TrimWhitespace
        {
            get { return trimWhitespace; }
            set { trimWhitespace = value; }
        }
        /// <summary>
        /// Get or set whether to skip blank lines.
        /// </summary>
        public bool SkipBlankLines
        {
            get { return skipBlankLines; }
            set { skipBlankLines = value; }
        }
        /// <summary>
        /// Get or set whether to skip lines with all columns blank.
        /// </summary>
        public bool SkipBlankColumnsLines
        {
            get { return skipBlankColumnsLines; }
            set { skipBlankColumnsLines = value; }
        }
        /// <summary>
        /// Get the number of the current row.
        /// </summary>
        public long RowNumber
        {
            get { return rowNumber; }
        }

        /// <summary>
        /// Create a new SpreadsheetReader.
        /// </summary>
        /// <param name="fname">the source of the spreadsheet</param>
        public SpreadsheetReader(string fname)
        {
            filename = fname;
            // determine format:
            xlsFormat = false;
            //Console.WriteLine("Extension: " + Path.GetExtension(filename).ToLower());
            if (Path.GetExtension(filename).ToLower() == ".xls")
            {
                xlsFormat = true;
            }
            // construct connection string:
            string connString = "Provider=Microsoft.Jet.OLEDB.4.0;" +
                "Data Source=";
            if (xlsFormat) // Excel Format
            {
                connString += "\"" + Path.GetFullPath(filename) + "\"" + ";" +
                    //	"Extended Properties=\"Excel 8.0;HDR=No\"";
                    "Extended Properties=\"Excel 8.0;HDR=Yes\"";
            }
            else  // assume CSV
            {
                connString += Path.GetDirectoryName(Path.GetFullPath(filename)) + ";" +
                    //	"Extended Properties=\"text;HDR=No;FMT=Delimited\"";
                    "Extended Properties=\"text;HDR=Yes;FMT=Delimited\"";
            }
            //Console.WriteLine("Opening with: " + connString);
            string selectString = "SELECT * FROM ";
            if (xlsFormat) // Excel Format
            {
                selectString += "[Sheet1$]";  // hard-coded first sheet? ***
                //selectString += "foo";  // hard-coded first sheet? ***
            }
            else  // assume CSV
            {
                selectString += Path.GetFileName(filename);
            }
            //Console.WriteLine("Selecting with: " + selectString);

            oleConn = new OleDbConnection(connString);
            oleConn.Open();
            //OleDbCommand openCmd = new OleDbCommand(selectString, oleConn);
            OleDbCommand openCmd = oleConn.CreateCommand();
            openCmd.CommandText = selectString;
            //openCmd.Connection.Open();
            //Console.WriteLine("Opened connection.");
            //Console.WriteLine("Database:  " + oleConn.Database);
            //Console.WriteLine("Datasource:  " + oleConn.DataSource);

            // start up reader:
            //oleReader = openCmd.ExecuteReader(CommandBehavior.CloseConnection);

            //MessageBox.Show(connString, "Excel Connection String");
            //MessageBox.Show(selectString, "Excel Select String");
            oleReader = openCmd.ExecuteReader();
            //Console.WriteLine("Started Reader.");

            //TestConnection();

            // Initialize the row reading:
            isEof = !Next();
            // Initialize the column names:
            SetupColumnNames();
        }

        /// <summary>
        /// Test the connection to the spreadsheet.
        /// </summary>
        public void TestConnection()
        {
            if (oleReader == null)
            {
                Console.WriteLine("OleDbDataReader is not initialized!");
                return;
            }
            // display column names
            for (int c = 0; c < oleReader.FieldCount; c++)
            {
                Console.Write(oleReader.GetName(c) + " \t");
            }
            Console.WriteLine("");
            Console.WriteLine("--------------------------------------------------");
            // display data
            while (oleReader.Read())
            {
                object[] fields = new Object[oleReader.FieldCount];
                oleReader.GetValues(fields);
                for (int i = 0; i < fields.Length; i++)
                {
                    Console.Write("" + fields[i] + " \t");
                }
                Console.WriteLine("");
            }
        }

        /// <summary>
        /// Read an entire row and advance the reader.
        /// </summary>
        /// <returns>The current row as an array of fields</returns>
        public object[] ReadRowObjects()
        {
            if (Eof())
                return new Object[0];  // should it be a null?
            object[] fields = new Object[oleReader.FieldCount];
            oleReader.GetValues(fields);
            for (int i = 0; i < fields.Length; i++)
            {
                if (fields[i] == DBNull.Value)
                {
                    fields[i] = null;
                }
            }
            Next();
            return fields;
        }

        /// <summary>
        /// Read an entire row and advance the reader.
        /// </summary>
        /// <returns>The current row as an array of fields</returns>
        public string[] ReadRow()
        {
            if (Eof())
                return new string[0];  // should it be a null?
            object[] fields = new Object[oleReader.FieldCount];
            string[] fieldsStr = new string[oleReader.FieldCount];
            oleReader.GetValues(fields);
            for (int i = 0; i < fields.Length; i++)
            {
                if (fields[i] == DBNull.Value)
                {
                    fieldsStr[i] = "";
                }
                else
                {
                    fieldsStr[i] = fields[i].ToString();
                }
            }
            Next();
            return fieldsStr;
        }

        //// Not certain if the FieldCount is stable. ***
        private void SetupColumnNames()
        {
            if (oleReader.FieldCount <= 0)
            {
                cols = new string[0];
                return;
            }
            cols = new string[oleReader.FieldCount];
            for (int c = 0; c < oleReader.FieldCount; c++)
            {
                cols[c] = oleReader.GetName(c);
                if (cols[c] == null)
                {
                    cols[c] = "";
                }
            }
        }

        /// <summary>
        /// Get the names of the columns.
        /// </summary>
        public string[] ColumnNames
        {
            get
            {
                return cols;
            }
        }

        /// <summary>
        /// Get the field at the column index.
        /// </summary>
        public object this[int i]
        {
            get
            {
                if (Eof())
                    return null;
                object item = oleReader[i];
                if (item == DBNull.Value)
                    return null;
                if (TrimWhitespace)
                {
                    if (item.GetType() == typeof(String))
                    {
                        item = ((string)item).Trim();
                    }
                }
                return item;
            }
        }

        /// <summary>
        /// Set the field at the column with the given header.
        /// </summary>
        public object this[string col]
        {
            get
            {
                if (Eof())
                    return null;
                object item = oleReader[col];
                if (item == DBNull.Value)
                    return null;
                if (TrimWhitespace)
                {
                    if (item.GetType() == typeof(String))
                    {
                        item = ((string)item).Trim();
                    }
                }
                return item;
            }
        }

        /// <summary>
        /// Get the field at the column index.
        /// </summary>
        public string this[int i, bool b]
        {
            get
            {
                object item = this[i];
                if (item == null)
                    return "";
                string res = item.ToString();
                if (TrimWhitespace)
                {
                    res = res.Trim();
                }
                return res;
            }
        }

        /// <summary>
        /// Set the field at the column with the given header.
        /// </summary>
        public string this[string col, bool b]
        {
            get
            {
                object item = this[col];
                if (item == null)
                    return "";
                string res = item.ToString();
                if (TrimWhitespace)
                {
                    res = res.Trim();
                }
                return res;
            }
        }

        /// <summary>
        /// Advance to the next row.
        /// </summary>
        /// <returns>true if there are more rows, false if at end of table</returns>
        public bool Next()
        {
            if (Eof() || oleReader == null)
                return false;
            while (true)
            {
                rowNumber++;
                isEof = !oleReader.Read();
                if (isEof)
                    break;
                if (!SkipBlankLines && !SkipBlankColumnsLines)
                    break;
                if (oleReader.FieldCount == 0)
                    continue;  // does this happen?
                int col;
                for (col = 0; col < RowLength(); col++)
                {
                    if (SkipBlankColumnsLines)
                    {
                        if (this[col, true] != "")
                            break;
                    }
                    else
                    {
                        if (this[col] != null)
                            break;
                    }
                }
                if (col < RowLength())
                    break;
            }
            return !isEof;
        }

        /// <summary>
        /// Get the number of fields in the current row.
        /// </summary>
        /// <returns>the number of fields in the current row</returns>
        public int RowLength()
        {
            if (Eof())
                return 0;
            return oleReader.FieldCount;
        }

        /// <summary>
        /// Check for end of file.
        /// </summary>
        /// <returns>true if at end of file, false otherwise</returns>
        public bool Eof()
        {
            return isEof;
        }

        /// <summary>
        /// Close the table.
        /// </summary>
        public void Close()
        {
            if (oleReader != null)
                oleReader.Close();
            oleReader = null;
            if (oleConn != null)
                oleConn.Close();
            oleConn = null;
            isEof = true;
        }
    }

    //// TODO: ODBC driver support?
    ////       Bare CsvReader support / merging?
    ////       Decent exception handling
    /// <summary>
    /// Write spreadsheet data in various formats.
    /// </summary>
    public class SpreadsheetWriter
    {
        private string filename;
        private bool xlsFormat;
        private OleDbConnection oleConn;
        //private OleDbDataAdapter oleAdapter;
        //private DataTable dataTable;
        //private string[] cols;

        // trim whitespace from each entry:
        private bool trimWhitespace = true;

        /// <summary>
        /// Gets or sets whether to trim whitespace from each field.
        /// </summary>
        public bool TrimWhitespace
        {
            get { return trimWhitespace; }
            set { trimWhitespace = value; }
        }

        /// <summary>
        /// Create a new SpreadsheetWriter.
        /// </summary>
        /// <param name="fname">the source of the spreadsheet</param>
        public SpreadsheetWriter(string fname)
        {
            filename = fname;

            // determine format:
            xlsFormat = false;
            //Console.WriteLine("Extension: " + Path.GetExtension(filename).ToLower());
            if (Path.GetExtension(filename).ToLower() == ".xls")
            {
                xlsFormat = true;
            }

            // construct connection string:
            string connString = "Provider=Microsoft.Jet.OLEDB.4.0;" +
                "Data Source=";
            if (xlsFormat) // Excel Format
            {
                connString += filename + ";" +
                    //	"Extended Properties=Excel 8.0;";
                    "Extended Properties=\"Excel 8.0;HDR=No\"";
            }
            else  // assume CSV
            {
                connString += Path.GetDirectoryName(Path.GetFullPath(filename)) + ";" +
                    "Extended Properties=\"text;HDR=Yes;FMT=Delimited\"";
            }
            //Console.WriteLine("Opening with: " + connString);

            string selectString = "SELECT * FROM ";
            if (xlsFormat) // Excel Format
            {
                selectString += "[Sheet1$]";  // hard-coded first sheet? ***
            }
            else  // assume CSV
            {
                selectString += Path.GetFileName(filename);
            }

            //// open the connection:
            oleConn = new OleDbConnection(connString);
            oleConn.Open();

            //oleAdapter = new OleDbDataAdapter(selectString, connString);
            //oleAdapter = new OleDbDataAdapter(selectString, oleConn);
            //OleDbCommandBuilder oleCommandBuilder = new OleDbCommandBuilder(oleAdapter);
            //oleConn = oleAdapter.SelectCommand.Connection;
            //oleConn.Open();
            //dataTable = new DataTable("sheet");

            //oleAdapter.InsertCommand = new OleDbCommand();
            //oleAdapter.InsertCommand.CommandText = "INSERT INTO [Sheet1$]";
            //// VALUES ('Other','Figimingle','c:\\images\\gardenhose.bmp')";
            //oleAdapter.InsertCommand.Connection = oleConn;

            //OleDbCommand openCmd = new OleDbCommand(selectString, oleConn);
            ////OleDbCommand openCmd = oleConn.CreateCommand();
            ////openCmd.CommandText = selectString;
            //openCmd.Connection.Open();
            //Console.WriteLine("Opened connection.");
            //Console.WriteLine("Database:  " + oleConn.Database);
            //Console.WriteLine("Datasource:  " + oleConn.DataSource);

            // start up reader:
            //oleReader = openCmd.ExecuteReader(CommandBehavior.CloseConnection);
            ////oleReader = openCmd.ExecuteReader();
            //Console.WriteLine("Started Reader.");

            //TestConnection();

            // Initialize the column names:
            //SetupColumnNames();
        }

        private string sqlString(string item)
        {
            return "'" + item.Replace("'", "''") + "'";
        }

        private string columnNameTrue(int col)
        {
            int aInt = (int)'A';
            int zInt = (int)'Z';
            int range = (zInt - aInt + 1);

            string res = "";
            int high = col / range;
            if (high > 0)
            {
                col = col % range;
                res += (char)(aInt + high - 1);
            }
            res += (char)(aInt + col);
            return res;
        }

        private string columnName(int col)
        {
            string res = "F";
            res += (col + 1);
            return res;
        }

        /// <summary>
        /// Write an entire row and advance the writer.
        /// </summary>
        /// <param name="fields">The row to write as an array of fields</param>
        public void WriteRow(object[] fields)
        {
            // widen if needed:
            //while (fields.Length > dataTable.Columns.Count)
            //{
            //	DataColumn col = new DataColumn();
            //	dataTable.Columns.Add(col);
            //}
            OleDbCommand oleCmd = new OleDbCommand();
            oleCmd.Connection = oleConn;
            string cmd = "INSERT INTO " + "[Sheet1$] "; // + "(FirstName, LastName) ";

            // give column names:
            cmd += "(";
            for (int i = 0; i < fields.Length; i++)
            {
                if (i > 0)
                {
                    cmd += ", ";
                }
                cmd += columnName(i);
            }
            cmd += ") ";

            cmd += "VALUES (";
            for (int i = 0; i < fields.Length; i++)
            {
                if (i > 0)
                {
                    cmd += ", ";
                }
                cmd += sqlString(fields[i].ToString());
            }
            //'Bill', 'Brown'
            cmd += ")";
            oleCmd.CommandText = cmd;
            try
            {
                oleCmd.ExecuteNonQuery();
            }
            catch (Exception)
            {
                throw new Exception("Excel insert error.");
            }

            //if (oleAdapter == null)  return;
            // widen if needed:
            //while (fields.Length > dataTable.Columns.Count)
            //{
            //	DataColumn col = new DataColumn();
            //	dataTable.Columns.Add(col);
            //}
            //dataTable.LoadDataRow(fields, false);
            //for (int i = 0; i < fields.Length; i++)
            //{
            //	if (fields[i] == DBNull.Value)
            //	{
            //		fields[i] = null;
            //	}
            //}
        }

        /// <summary>
        /// Get the number of fields in the current row.
        /// </summary>
        /// <returns>the number of fields in the current row</returns>
        public int RowLength()
        {
            return 0;
            //return dataTable.Columns.Count;
        }

        /// <summary>
        /// Close the table.
        /// </summary>
        public void Close()
        {
            //if (oleAdapter != null)
            //{
            //	oleAdapter.Update(dataTable);
            //	oleAdapter = null;
            //}
            if (oleConn != null)
            {
                oleConn.Close();
                oleConn = null;
            }
        }
    }
#endif
}

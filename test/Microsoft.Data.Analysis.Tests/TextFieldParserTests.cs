// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Text;
using Xunit;

namespace Microsoft.Data.Analysis.Tests
{
    public class TextFieldParserTests
    {
        private Stream GetStream(string streamData)
        {
            return new MemoryStream(Encoding.Default.GetBytes(streamData));
        }

        [Fact]
        public void Constructors()
        {
            string data = @"abc123";

            // public TextFieldParser(System.IO.Stream stream)
            using (var stream = GetStream(data))
            {
                using (var parser = new TextFieldParser(stream))
                {
                }
                Assert.Throws<ObjectDisposedException>(() => stream.ReadByte());
            }

            // public TextFieldParser(System.IO.Stream stream, System.Text.Encoding defaultEncoding, bool detectEncoding, bool leaveOpen);
            using (var stream = GetStream(data))
            {
                using (var parser = new TextFieldParser(stream, defaultEncoding: Encoding.Unicode, detectEncoding: true, leaveOpen: true))
                {
                }
                _ = stream.ReadByte();
            }

            // public TextFieldParser(System.IO.TextReader reader)
            using (var reader = new StreamReader(GetStream(data)))
            {
                using (var parser = new TextFieldParser(reader))
                {
                }
                Assert.Throws<ObjectDisposedException>(() => reader.ReadToEnd());
            }
        }

        [Fact]
        public void Close()
        {
            string data = @"abc123";
            using (var stream = GetStream(data))
            {
                using (var parser = new TextFieldParser(stream))
                {
                    parser.Close();
                }
            }

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                parser.Close();
            }

            {
                var parser = new TextFieldParser(GetStream(data));
                parser.Close();
                parser.Close();
            }

            {
                TextFieldParser parser;
                using (parser = new TextFieldParser(GetStream(data)))
                {
                }
                parser.Close();
            }
        }

        [Fact]
        public void Properties()
        {

            string data = @"abc123";

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                Assert.Null(parser.CommentTokens);
                parser.CommentTokens = new[] { "[", "]" };
                Assert.Equal(new[] { "[", "]" }, parser.CommentTokens);

                Assert.Null(parser.Delimiters);
                parser.SetDelimiters(new[] { "A", "123" });
                Assert.Equal(new[] { "A", "123" }, parser.Delimiters);
                parser.SetDelimiters(new[] { "123", "B" });
                Assert.Equal(new[] { "123", "B" }, parser.Delimiters);

                Assert.Null(parser.FieldWidths);
                parser.SetFieldWidths(new[] { 1, 2, int.MaxValue });
                Assert.Equal(new[] { 1, 2, int.MaxValue }, parser.FieldWidths);
                parser.SetFieldWidths(new[] { int.MaxValue, 3 });
                Assert.Equal(new[] { int.MaxValue, 3 }, parser.FieldWidths);
                Assert.Throws<ArgumentException>(() => parser.SetFieldWidths(new[] { -1, -1 }));

                Assert.True(parser.HasFieldsEnclosedInQuotes);
                parser.HasFieldsEnclosedInQuotes = false;
                Assert.False(parser.HasFieldsEnclosedInQuotes);

                Assert.Equal(FieldType.Delimited, parser.TextFieldType);
                parser.TextFieldType = FieldType.FixedWidth;
                Assert.Equal(FieldType.FixedWidth, parser.TextFieldType);

                Assert.True(parser.TrimWhiteSpace);
                parser.TrimWhiteSpace = false;
                Assert.False(parser.TrimWhiteSpace);
            }
        }

        [Fact]
        public void ReadLine_ReadToEnd()
        {
            string data = @"abc
123";

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                Assert.False(parser.EndOfData);

                Assert.Equal(
@"abc
123",
                    parser.ReadToEnd());
                Assert.Equal(-1, parser.LineNumber);
                Assert.True(parser.EndOfData);
            }

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                Assert.Equal("abc", parser.ReadLine());
                Assert.Equal(2, parser.LineNumber);
                Assert.False(parser.EndOfData);

                Assert.Equal("123", parser.ReadToEnd());
                Assert.Equal(-1, parser.LineNumber);
                Assert.True(parser.EndOfData);
            }

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                Assert.Equal("abc", parser.ReadLine());
                Assert.Equal(2, parser.LineNumber);
                Assert.False(parser.EndOfData);

                Assert.Equal("123", parser.ReadLine());
                Assert.Equal(-1, parser.LineNumber);
                Assert.True(parser.EndOfData);

                Assert.Null(parser.ReadToEnd());
                Assert.Equal(-1, parser.LineNumber);
                Assert.True(parser.EndOfData);
            }
        }

        [Fact]
        public void ErrorLine()
        {
            string data = @"abc 123
def 45
ghi 789";

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                parser.TextFieldType = FieldType.FixedWidth;
                parser.SetFieldWidths(new[] { 3, 4 });

                Assert.Equal(-1, parser.ErrorLineNumber);
                Assert.Equal("", parser.ErrorLine);

                Assert.Equal(new[] { "abc", "123" }, parser.ReadFields());
                Assert.Equal(-1, parser.ErrorLineNumber);
                Assert.Equal("", parser.ErrorLine);

                Assert.Throws<Exception>(() => parser.ReadFields());
                Assert.Equal(2, parser.ErrorLineNumber);
                Assert.Equal("def 45", parser.ErrorLine);

                Assert.Equal(new[] { "ghi", "789" }, parser.ReadFields());
                Assert.Equal(2, parser.ErrorLineNumber);
                Assert.Equal("def 45", parser.ErrorLine);
            }
        }

        [Fact]
        public void HasFieldsEnclosedInQuotes_TrimWhiteSpace()
        {
            string data = @""""", "" "" ,""abc"", "" 123 "" ,";

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                parser.SetDelimiters(new[] { "," });
                Assert.Equal(new[] { "", "", "abc", "123", "" }, parser.ReadFields());
            }

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                parser.TrimWhiteSpace = false;
                parser.SetDelimiters(new[] { "," });
                Assert.Equal(new[] { "", " ", "abc", " 123 ", "" }, parser.ReadFields());
            }

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                parser.HasFieldsEnclosedInQuotes = false;
                parser.SetDelimiters(new[] { "," });
                Assert.Equal(new[] { @"""""", @""" """, @"""abc""", @""" 123 """, "" }, parser.ReadFields());
            }

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                parser.TrimWhiteSpace = false;
                parser.HasFieldsEnclosedInQuotes = false;
                parser.SetDelimiters(new[] { "," });
                Assert.Equal(new[] { @"""""", @" "" "" ", @"""abc""", @" "" 123 "" ", "" }, parser.ReadFields());
            }

            data = @""","", "", """;
            using (var parser = new TextFieldParser(GetStream(data)))
            {
                parser.TrimWhiteSpace = false;
                parser.SetDelimiters(new[] { "," });
                Assert.Equal(new[] { ",", ", " }, parser.ReadFields());
            }
        }

        [Fact]
        public void PeekChars()
        {
            string data = @"abc,123
def,456
ghi,789";

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                Assert.Throws<ArgumentException>(() => parser.PeekChars(0));

                Assert.Equal("a", parser.PeekChars(1));
                Assert.Equal("abc,123", parser.PeekChars(10));

                Assert.Equal("abc,123", parser.ReadLine());

                parser.TextFieldType = FieldType.FixedWidth;
                parser.SetFieldWidths(new[] { 3, -1 });

                Assert.Equal("d", parser.PeekChars(1));
                Assert.Equal("def,456", parser.PeekChars(10));
                Assert.Equal(new[] { "def", ",456" }, parser.ReadFields());

                parser.TextFieldType = FieldType.Delimited;
                parser.SetDelimiters(new[] { "," });

                Assert.Equal("g", parser.PeekChars(1));
                Assert.Equal("ghi,789", parser.PeekChars(10));
                Assert.Equal(new[] { "ghi", "789" }, parser.ReadFields());

                Assert.Null(parser.PeekChars(1));
                Assert.Null(parser.PeekChars(10));
            }
        }

        [Fact]
        public void ReadFields_FieldWidths()
        {
            string data = @"abc,123
def,456
ghi,789";

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                parser.TextFieldType = FieldType.FixedWidth;

                Assert.Throws<InvalidOperationException>(() => parser.ReadFields());

                parser.SetFieldWidths(new[] { -1 });
                Assert.Equal(new[] { "abc,123" }, parser.ReadFields());

                parser.SetFieldWidths(new[] { 3, -1 });
                Assert.Equal(new[] { "def", ",456" }, parser.ReadFields());

                parser.SetFieldWidths(new[] { 3, 2 });
                Assert.Equal(new[] { "ghi", ",7" }, parser.ReadFields());

                parser.SetFieldWidths(new[] { 3, 2 });
                Assert.Null(parser.ReadFields());
            }
        }

        [Fact]
        public void ReadFields_Delimiters_LineNumber()
        {
            string data = @"abc,123
def,456
ghi,789";

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                Assert.Equal(1, parser.LineNumber);

                Assert.Throws<Exception>(() => parser.ReadFields());
                Assert.Equal(1, parser.LineNumber);

                parser.SetDelimiters(new[] { "," });
                Assert.Equal(new[] { "abc", "123" }, parser.ReadFields());
                Assert.Equal(2, parser.LineNumber);

                parser.SetDelimiters(new[] { ";", "," });
                Assert.Equal(new[] { "def", "456" }, parser.ReadFields());
                Assert.Equal(3, parser.LineNumber);

                parser.SetDelimiters(new[] { "g", "9" });
                Assert.Equal(new[] { "", "hi,78", "" }, parser.ReadFields());
                Assert.Equal(-1, parser.LineNumber);
            }

            data = @",,

,
";

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                Assert.Equal(1, parser.LineNumber);

                parser.SetDelimiters(new[] { "," });
                Assert.Equal(new[] { "", "", "" }, parser.ReadFields());
                Assert.Equal(2, parser.LineNumber);

                // ReadFields should ignore the empty new line
                Assert.Equal(new[] { "", "" }, parser.ReadFields());
                Assert.Equal(-1, parser.LineNumber);

                Assert.Null(parser.ReadFields());
                Assert.Equal(-1, parser.LineNumber);

                Assert.Null(parser.ReadFields());
                Assert.Equal(-1, parser.LineNumber);
            }
        }

        [Fact]
        public void UnmatchedQuote_MalformedLineException()
        {
            string data = @""""", """;

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                parser.SetDelimiters(new[] { "," });
                Assert.Throws<Exception>(() => parser.ReadFields());
            }
        }

        [Fact]
        public void ReadFields_QuoteOnNewLine()
        {
            string data = @"abc,""123
123""
def,456
ghi,789";

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                Assert.Equal(1, parser.LineNumber);

                Assert.Throws<Exception>(() => parser.ReadFields());
                Assert.Equal(1, parser.LineNumber);

                parser.SetDelimiters(new[] { "," });
                Assert.Equal(new[] { "abc", @"123
123" }, parser.ReadFields());
                Assert.Equal(3, parser.LineNumber);

                parser.SetDelimiters(new[] { ";", "," });
                Assert.Equal(new[] { "def", "456" }, parser.ReadFields());
                Assert.Equal(4, parser.LineNumber);

                parser.SetDelimiters(new[] { "g", "9" });
                Assert.Equal(new[] { "", "hi,78", "" }, parser.ReadFields());
                Assert.Equal(-1, parser.LineNumber);
            }

            data = @",,

,
";

            using (var parser = new TextFieldParser(GetStream(data)))
            {
                Assert.Equal(1, parser.LineNumber);

                parser.SetDelimiters(new[] { "," });
                Assert.Equal(new[] { "", "", "" }, parser.ReadFields());
                Assert.Equal(2, parser.LineNumber);

                // ReadFields should ignore the empty new line
                Assert.Equal(new[] { "", "" }, parser.ReadFields());
                Assert.Equal(-1, parser.LineNumber);

                Assert.Null(parser.ReadFields());
                Assert.Equal(-1, parser.LineNumber);

                Assert.Null(parser.ReadFields());
                Assert.Equal(-1, parser.LineNumber);
            }
        }
    }
}

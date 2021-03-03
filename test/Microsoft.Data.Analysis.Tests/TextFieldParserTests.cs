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
    }
}

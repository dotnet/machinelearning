// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Text;
using Xunit.Abstractions;

namespace Microsoft.ML.TestFrameworkCommon
{
    public sealed class TestLogger : TextWriter
    {
        private Encoding _encoding;
        private ITestOutputHelper _testOutput;

        public override Encoding Encoding => _encoding;

        public TestLogger(ITestOutputHelper testOutput)
        {
            _testOutput = testOutput;
            _encoding = new UnicodeEncoding();
        }

        public override void Write(char value)
        {
            _testOutput.WriteLine($"{value}");
        }

        public override void Write(string value)
        {
            if (value.EndsWith("\r\n"))
                value = value.Substring(0, value.Length - 2);
            _testOutput.WriteLine(value);
        }

        public override void Write(string format, params object[] args)
        {
            if (format.EndsWith("\r\n"))
                format = format.Substring(0, format.Length - 2);

            _testOutput.WriteLine(format, args);
        }

        public override void Write(char[] buffer, int index, int count)
        {
            var span = buffer.AsSpan(index, count);
            if ((span.Length >= 2) && (span[count - 2] == '\r') && (span[count - 1] == '\n'))
                span = span.Slice(0, count - 2);
            _testOutput.WriteLine(span.ToString());
        }
    }
}

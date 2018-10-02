// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.TestFramework
{
    /// <summary>
    /// On demand open up new streams over a source of bytes.
    /// </summary>
    public sealed class BytesStreamSource : IMultiStreamSource
    {
        private readonly byte[] _data;

        public int Count => 1;

        public BytesStreamSource(byte[] data)
        {
            Contracts.AssertValue(data);
            _data = data;
        }

        public BytesStreamSource(string data)
            : this(Encoding.UTF8.GetBytes(data))
        {
        }

        public string GetPathOrNull(int index)
        {
            Contracts.Check(index == 0);
            return null;
        }

        public Stream Open(int index)
        {
            Contracts.Check(index == 0);
            return new MemoryStream(_data, writable: false);
        }

        public TextReader OpenTextReader(int index)
        {
            return new StreamReader(Open(index));
        }
    }
}

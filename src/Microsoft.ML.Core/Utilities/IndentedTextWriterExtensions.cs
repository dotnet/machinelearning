// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CodeDom.Compiler;
using System.IO;
using System.Text;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    [BestFriend]
    internal static class IndentedTextWriterExtensions
    {
        public struct Scope : IDisposable
        {
            private IndentedTextWriter _writer;

            public Scope(IndentedTextWriter writer)
            {
                _writer = writer;
                _writer.Indent();
            }
            public void Dispose()
            {
                _writer.Outdent();
                _writer = null;
            }
        }

        public static Scope Nest(this IndentedTextWriter writer)
        {
            return new Scope(writer);
        }

        public static void Indent(this IndentedTextWriter writer)
        {
            writer.Indent++;
        }
        public static void Outdent(this IndentedTextWriter writer)
        {
            writer.Indent--;
        }

        public static void WriteLineNoTabs(this IndentedTextWriter writer)
        {
            writer.WriteLineNoTabs(string.Empty);
        }
    }
}
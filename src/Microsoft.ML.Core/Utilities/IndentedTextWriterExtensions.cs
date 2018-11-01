// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CodeDom.Compiler;
using System.IO;
using System.Text;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    public static class IndentedTextWriterExtensions
    {
        public struct Scope : IDisposable
        {
            private IndentedTextWriter _wrt;

            public Scope(IndentedTextWriter wrt)
            {
                _wrt = wrt;
                _wrt.Indent();
            }
            public void Dispose()
            {
                _wrt.Outdent();
                _wrt = null;
            }
        }

        public static Scope Nest(this IndentedTextWriter wrt)
        {
            return new Scope(wrt);
        }

        public static void Indent(this IndentedTextWriter wrt)
        {
            wrt.Indent++;
        }
        public static void Outdent(this IndentedTextWriter wrt)
        {
            --wrt.Indent;
        }
    }
}
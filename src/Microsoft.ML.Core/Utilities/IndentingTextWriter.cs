// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Text;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    [BestFriend]
    internal sealed class IndentingTextWriter : TextWriter
    {
        public struct Scope : IDisposable
        {
            private IndentingTextWriter _wrt;

            public Scope(IndentingTextWriter wrt)
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

        private readonly TextWriter _wrt;
        private readonly string _str;

        private bool _bol;
        private int _indent;
        private bool _skipLine;

        public static IndentingTextWriter Wrap(TextWriter wrt, string indentString = "  ")
        {
            Contracts.AssertValue(wrt);
            Contracts.AssertNonEmpty(indentString);

            IndentingTextWriter itw = wrt as IndentingTextWriter;
            if (itw != null)
                return itw;
            return new IndentingTextWriter(wrt, indentString);
        }

        private IndentingTextWriter(TextWriter wrt, string indentString)
        {
            Contracts.AssertValue(wrt);
            _wrt = wrt;
            _str = indentString;
            _bol = true;
        }

        public Scope Nest()
        {
            return new Scope(this);
        }

        public void Indent()
        {
            _indent++;
        }

        public void Outdent()
        {
            Contracts.Assert(_indent > 0);
            --_indent;
            _skipLine = false;
        }

        public void Skip()
        {
            _skipLine = true;
        }

        private void Adjust()
        {
            if (_bol)
            {
                if (_skipLine)
                {
                    _wrt.WriteLine();
                    _skipLine = false;
                }
                for (int i = 0; i < _indent; i++)
                    _wrt.Write(_str);
                _bol = false;
            }
        }

        private void AdjustLine()
        {
            Adjust();
            _bol = true;
        }

        public override System.Text.Encoding Encoding
        {
            get { return _wrt.Encoding; }
        }

        public override void Write(bool value)
        {
            Adjust();
            _wrt.Write(value);
        }
        public override void Write(char value)
        {
            Adjust();
            _wrt.Write(value);
        }
        public override void Write(char[] buffer)
        {
            Adjust();
            _wrt.Write(buffer);
        }
        public override void Write(decimal value)
        {
            Adjust();
            _wrt.Write(value);
        }
        public override void Write(double value)
        {
            Adjust();
            _wrt.Write(value);
        }
        public override void Write(float value)
        {
            Adjust();
            _wrt.Write(value);
        }
        public override void Write(int value)
        {
            Adjust();
            _wrt.Write(value);
        }
        public override void Write(long value)
        {
            Adjust();
            _wrt.Write(value);
        }
        public override void Write(object value)
        {
            Adjust();
            _wrt.Write(value);
        }
        public override void Write(string value)
        {
            Adjust();
            _wrt.Write(value);
        }
        public override void Write(uint value)
        {
            Adjust();
            _wrt.Write(value);
        }
        public override void Write(ulong value)
        {
            Adjust();
            _wrt.Write(value);
        }
        public override void Write(string format, object arg0)
        {
            Adjust();
            _wrt.Write(format, arg0);
        }
        public override void Write(string format, params object[] arg)
        {
            Adjust();
            _wrt.Write(format, arg);
        }
        public override void Write(char[] buffer, int index, int count)
        {
            Adjust();
            _wrt.Write(buffer, index, count);
        }
        public override void Write(string format, object arg0, object arg1)
        {
            Adjust();
            _wrt.Write(format, arg0, arg1);
        }
        public override void Write(string format, object arg0, object arg1, object arg2)
        {
            Adjust();
            _wrt.Write(format, arg0, arg1, arg2);
        }

        public override void WriteLine()
        {
            _bol = true;
            _skipLine = false;
            _wrt.WriteLine();
        }
        public override void WriteLine(bool value)
        {
            AdjustLine();
            _wrt.WriteLine(value);
        }
        public override void WriteLine(char value)
        {
            AdjustLine();
            _wrt.WriteLine(value);
        }
        public override void WriteLine(char[] buffer)
        {
            AdjustLine();
            _wrt.WriteLine(buffer);
        }
        public override void WriteLine(decimal value)
        {
            AdjustLine();
            _wrt.WriteLine(value);
        }
        public override void WriteLine(double value)
        {
            AdjustLine();
            _wrt.WriteLine(value);
        }
        public override void WriteLine(float value)
        {
            AdjustLine();
            _wrt.WriteLine(value);
        }
        public override void WriteLine(int value)
        {
            AdjustLine();
            _wrt.WriteLine(value);
        }
        public override void WriteLine(long value)
        {
            AdjustLine();
            _wrt.WriteLine(value);
        }
        public override void WriteLine(object value)
        {
            AdjustLine();
            _wrt.WriteLine(value);
        }
        public override void WriteLine(string value)
        {
            AdjustLine();
            _wrt.WriteLine(value);
        }
        public override void WriteLine(uint value)
        {
            AdjustLine();
            _wrt.WriteLine(value);
        }
        public override void WriteLine(ulong value)
        {
            AdjustLine();
            _wrt.WriteLine(value);
        }
        public override void WriteLine(string format, object arg0)
        {
            AdjustLine();
            _wrt.WriteLine(format, arg0);
        }
        public override void WriteLine(string format, params object[] arg)
        {
            AdjustLine();
            _wrt.WriteLine(format, arg);
        }
        public override void WriteLine(char[] buffer, int index, int count)
        {
            AdjustLine();
            _wrt.WriteLine(buffer, index, count);
        }
        public override void WriteLine(string format, object arg0, object arg1)
        {
            AdjustLine();
            _wrt.WriteLine(format, arg0, arg1);
        }
        public override void WriteLine(string format, object arg0, object arg1, object arg2)
        {
            AdjustLine();
            _wrt.WriteLine(format, arg0, arg1, arg2);
        }
    }
}

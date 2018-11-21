// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.CommandLine
{
    public sealed class CharCursor
    {
        private readonly string _text;
        private readonly int _ichLim;

        private int _ichCur;
        private char _chCur;

        public bool Eof => _ichCur >= _ichLim;

        public int IchCur => _ichCur;
        public char ChCur => _chCur;

        public CharCursor(string text)
        {
            Contracts.CheckValueOrNull(text);
            _text = text;
            _ichLim = Utils.Size(text);
            _ichCur = 0;
            _chCur = _ichCur < _ichLim ? _text[_ichCur] : '\x00';
        }

        public CharCursor(string text, int min, int lim)
        {
            Contracts.CheckValueOrNull(text);
            Contracts.CheckParam(0 <= min && min <= Utils.Size(text), nameof(min));
            Contracts.CheckParam(min <= lim && lim <= Utils.Size(text), nameof(lim));
            _text = text;
            _ichLim = lim;
            _ichCur = min;
            _chCur = _ichCur < _ichLim ? _text[_ichCur] : '\x00';
        }

        public string GetRest()
        {
            return _ichCur < _ichLim ? _text.Substring(_ichCur) : "";
        }

        // Fetch the next character into _chCur and return it.
        public char ChNext()
        {
            if (++_ichCur < _ichLim)
                _chCur = _text[_ichCur];
            else
            {
                _ichCur = _ichLim;
                _chCur = '\x00';
            }
            return _chCur;
        }

        public char ChPeek(int dich)
        {
            Contracts.Assert(0 <= dich && dich <= _ichLim - _ichCur);

            int ich = dich + _ichCur;
            if (ich < _ichLim)
                return _text[ich];

            return '\x00';
        }
    }
}

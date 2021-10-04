// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    internal sealed class TokenCursor
    {
        // This is the token stream. We cache items as we consume them.
        // This code assumes that the enumerator will produce an Eof token,
        // When the Eof is produced, _tokens is disposed and set to null.
        private IEnumerator<Token> _tokens;

        // The cache buffer.
        private Token[] _buffer;

        // The logical token index of _buffer[0].
        private int _itokBase;

        // The limit of the cached range within _buffer (relative to the buffer).
        private int _itokLim;

        // The current position within the buffer.
        private int _itokCur;
        private Token _tokCur;
        private TokKind _tidCur;

        // If this is >= 0, this position is "pinned" - we need to keep it in the buffer.
        // This is to support rewinding.
        private int _itokPin;

        public TokenCursor(IEnumerable<Token> tokens)
        {
            Contracts.AssertValue(tokens);
            _tokens = tokens.GetEnumerator();
            _buffer = new Token[0x0400];
            _itokPin = -1;

            // Get the first token.
            FetchCore();

            _tokCur = _buffer[_itokCur];
            _tidCur = _tokCur.Kind;

            AssertValid();
        }

        [Conditional("DEBUG")]
        private void AssertValid()
        {
            Contracts.AssertValue(_buffer);

            // _itokCur should never reach _itokLim.
            Contracts.Assert(0 <= _itokCur && _itokCur < _itokLim && _itokLim <= _buffer.Length);

            // The last token in the buffer is Eof iff _tokens is null.
            Contracts.Assert((_tokens != null) == (_buffer[_itokLim - 1].Kind != TokKind.Eof));

            // _tokCur and _tidCur should match _itokCur.
            Contracts.Assert(_tokCur == _buffer[_itokCur]);
            Contracts.Assert(_tidCur == _tokCur.Kind);
        }

        public Token TokCur
        {
            get
            {
                AssertValid();
                return _tokCur;
            }
        }

        public TokKind TidCur
        {
            get
            {
                AssertValid();
                return _tidCur;
            }
        }

        public TokKind CtxCur
        {
            get
            {
                AssertValid();
                return _tokCur.KindContext;
            }
        }

        // This fetches an additional token from _tokens and caches it.
        // If needed, makes room in the cache (_buffer) by either sliding items
        // or resizing _buffer.
        private void FetchToken()
        {
            Contracts.Assert(_tokens != null);

            if (_itokLim >= _buffer.Length)
            {
                // Need more room. See if we can "slide".
                if (_itokCur > 0 && _itokPin != 0)
                {
                    int itokMin = _itokCur;
                    if (0 < _itokPin && _itokPin < itokMin)
                        itokMin = _itokPin;
                    int itokSrc = itokMin;
                    int itokDst = 0;
                    while (itokSrc < _itokLim)
                        _buffer[itokDst++] = _buffer[itokSrc++];
                    if (0 < _itokPin)
                        _itokPin -= itokMin;
                    _itokLim -= itokMin;
                    _itokCur -= itokMin;
                    _itokBase += itokMin;
                }
                else
                {
                    // Need to resize the buffer.
                    Array.Resize(ref _buffer, 2 * _buffer.Length);
                }
            }

            FetchCore();
        }

        private void FetchCore()
        {
            Contracts.Assert(_tokens != null);
            Contracts.Assert(_itokLim < _buffer.Length);

            if (!_tokens.MoveNext())
            {
                Contracts.Assert(false, "Token stream should end with an Eof token!");
                throw Contracts.Except();
            }

            // Cache the new token.
            Token tok = _buffer[_itokLim++] = _tokens.Current;
            Contracts.Assert(tok != null);

            // See if we're done pulling items from _tokens.
            if (tok.Kind == TokKind.Eof)
            {
                _tokens.Dispose();
                _tokens = null;
            }
        }

        // This expects that _itokCur + ditok is either within the buffered token range or
        // just at the end of it. In other words, it does not support skipping tokens.
        private void MoveBy(int ditok)
        {
            AssertValid();
            Contracts.Assert(-_itokCur <= ditok && ditok <= _itokLim - _itokCur);
            Contracts.Assert(ditok < _itokLim - _itokCur || _tokens != null);

            while (ditok >= _itokLim - _itokCur)
                FetchToken();

            _itokCur += ditok;
            _tokCur = _buffer[_itokCur];
            _tidCur = _tokCur.Kind;
            AssertValid();
        }

        public TokKind TidNext()
        {
            AssertValid();
            if (_tidCur != TokKind.Eof)
                MoveBy(1);
            return _tidCur;
        }

        /// <summary>
        /// This expects that ItokCur + ditok is either within the buffered token range or just
        /// at the end of it. In other words, it does not support skipping tokens.
        /// </summary>
        public Token TokPeek(int ditok)
        {
            AssertValid();
            Contracts.Assert(-_itokCur <= ditok && ditok <= _itokLim - _itokCur);
            Contracts.Assert(ditok < _itokLim - _itokCur || _tokens != null);

            while (ditok >= _itokLim - _itokCur)
                FetchToken();

            return _buffer[_itokCur + ditok];
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    internal sealed class Error
    {
        public readonly Token Token;
        public readonly string Message;
        // Args may be null.
        public readonly object[] Args;

        public Error(Token tok, string msg)
        {
            Contracts.AssertValue(tok);
            Contracts.AssertNonEmpty(msg);
            Token = tok;
            Message = msg;
            Args = null;
        }

        public Error(Token tok, string msg, params object[] args)
        {
            Contracts.AssertValue(tok);
            Contracts.AssertNonEmpty(msg);
            Contracts.AssertValue(args);
            Token = tok;
            Message = msg;
            Args = args;
        }

        public string GetMessage()
        {
            var msg = Message;
            if (Utils.Size(Args) > 0)
                msg = string.Format(msg, Args);
            if (Token != null)
                msg = string.Format("at '{0}': {1}", Token, msg);
            return msg;
        }
    }
}

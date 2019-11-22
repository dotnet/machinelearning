// Copyright (c) .NET Foundation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;

namespace Microsoft.DotNet.Cli.Utils
{
    internal static class ExceptionExtensions
    {
        public static TException DisplayAsError<TException>(this TException exception)
        where TException : Exception
        {
             exception.Data.Add(CliUserDisplayedException, true);
             return exception;
        }

        public static bool ShouldBeDisplayedAsError(this Exception e) =>
            e.Data.Contains(CliUserDisplayedException);

        internal const string CliUserDisplayedException = "CLI_User_Displayed_Exception";
    }
}

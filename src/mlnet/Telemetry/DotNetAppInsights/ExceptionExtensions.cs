// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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

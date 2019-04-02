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
             exception.Data.Add(CLI_User_Displayed_Exception, true);
             return exception;
        }

        public static bool ShouldBeDisplayedAsError(this Exception e) =>
            e.Data.Contains(CLI_User_Displayed_Exception);

        internal const string CLI_User_Displayed_Exception = "CLI_User_Displayed_Exception";
    }
}

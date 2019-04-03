// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.Globalization;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Auto
{
    internal static class Contracts
    {
        public static void Check(this IExceptionContext ctx, bool f, string msg)
        {
            if (!f)
            {
                throw Except(ctx, msg);
            }
        }

        public static void Check(this IExceptionContext ctx, bool f)
        {
            if (!f)
            {
                throw new InvalidOperationException();
            }
        }

        public static void CheckValue<T>(T val, string paramName) where T : class
        {
            if (object.ReferenceEquals(val, null))
            {
                throw new ArgumentNullException(paramName);
            }
        }

        public static void CheckValue<T>(this IExceptionContext ctx, T val, string paramName) where T : class
        {
            if (object.ReferenceEquals(val, null))
            {
                throw new ArgumentNullException(paramName);
            }
        }

        public static void CheckParam(this IExceptionContext ctx, bool f, string paramName)
        {
            if (!f)
            {
                throw ExceptParam(ctx, paramName);
            }
        }

        public static void CheckParam(bool f, string paramName)
        {
            if (!f)
            {
                throw ExceptParam(paramName);
            }
        }

        public static void Assert(bool f, string msg)
        {
            if (!f)
            {
                Debug.Fail(msg);
            }
        }

        public static Exception Except(this IExceptionContext ctx, string msg, params object[] args)
            => throw new InvalidOperationException(GetMsg(msg, args));

        public static Exception ExceptParam(this IExceptionContext ctx, string paramName)
            => new ArgumentOutOfRangeException(paramName);

        public static Exception Except(string msg) => new InvalidOperationException(msg);

        public static Exception ExceptParam(string paramName)
            => new ArgumentOutOfRangeException(paramName);

        private static string GetMsg(string msg, params object[] args)
        {
            try
            {
                msg = string.Format(CultureInfo.InvariantCulture, msg, args);
            }
            catch (FormatException ex)
            {
                Contracts.Assert(false, "Format string arg mismatch: " + ex.Message);
            }
            return msg;
        }
    }
}

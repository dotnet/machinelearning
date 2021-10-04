// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// We want the conditional code in this file to always be available to
// client assemblies that might be DEBUG versions. That is, if someone uses
// the release build of this assembly to build a DEBUG version of their code,
// we want Contracts.Assert to be fully functional for that client.

using System;
using System.Diagnostics;
using System.Globalization;

namespace Microsoft.ML.Internal.DataView
{
    internal static class Contracts
    {
        private static string GetMsg(string msg, params object[] args)
        {
            try
            {
                msg = string.Format(CultureInfo.InvariantCulture, msg, args);
            }
            catch (FormatException ex)
            {
                Debug.Fail("Format string arg mismatch: " + ex.Message);
                throw;
            }
            return msg;
        }

        public static Exception Except(string msg)
            => new InvalidOperationException(msg);

        public static Exception ExceptParam(string paramName)
            => new ArgumentOutOfRangeException(paramName);
        public static Exception ExceptParam(string paramName, string msg)
            => new ArgumentOutOfRangeException(paramName, msg);
        public static Exception ExceptParam(string paramName, string msg, params object[] args)
            => new ArgumentOutOfRangeException(paramName, GetMsg(msg, args));

        public static Exception ExceptValue(string paramName)
            => new ArgumentNullException(paramName);

        public static void Check(bool f, string msg)
        {
            if (!f)
                throw Except(msg);
        }

        public static void CheckParam(bool f, string paramName)
        {
            if (!f)
                throw ExceptParam(paramName);
        }
        public static void CheckParam(bool f, string paramName, string msg)
        {
            if (!f)
                throw ExceptParam(paramName, msg);
        }
        public static void CheckParam(bool f, string paramName, string msg, params object[] args)
        {
            if (!f)
                throw ExceptParam(paramName, msg, args);
        }

        public static void CheckValue<T>(T val, string paramName) where T : class
        {
            if (object.ReferenceEquals(val, null))
                throw ExceptValue(paramName);
        }
    }
}

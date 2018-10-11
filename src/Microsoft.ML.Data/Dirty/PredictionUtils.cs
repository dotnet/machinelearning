// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Internal.Internallearn
{
    using Float = System.Single;

    /// <summary>
    /// Various utilities
    /// </summary>
    public static class PredictionUtil
    {
        /// <summary>
        /// generic method for parsing arguments using CommandLine. If there's a problem, it throws an InvalidOperationException, with a message giving usage.
        /// </summary>
        /// <param name="env">The host environment</param>
        /// <param name="args">The argument object</param>
        /// <param name="settings">The settings string (for example, "threshold-")</param>
        /// <param name="name">The name is used for error reporting only</param>
        public static void ParseArguments(IHostEnvironment env, object args, string settings, string name = null)
        {
            if (string.IsNullOrWhiteSpace(settings))
                return;

            string errorMsg = null; // non-null errorMsg will indicate an error
            try
            {
                string err = null;
                string helpText;
                if (!CmdParser.ParseArguments(env, settings, args, e => { err = err ?? e; }, out helpText))
                    errorMsg = err + (!string.IsNullOrWhiteSpace(name) ? "\nUSAGE FOR '" + name + "':\n" : "\nUSAGE:\n") + helpText;
            }
            catch (Exception e)
            {
                Contracts.Assert(false);
                errorMsg = "Unexpected exception thrown while parsing: " + e.Message;
            }

            if (errorMsg != null)
                throw Contracts.Except(errorMsg);
        }

        // The extra settings are assumed to be "old style", so we apply the semi-colon hack to them.
        public static string CombineSettings(string[] settings, string[] extraSettings = null)
        {
            if (Utils.Size(extraSettings) == 0)
                return CmdParser.CombineSettings(settings);
            if (Utils.Size(settings) == 0)
                return CmdParser.CombineSettings(SplitOnSemis(extraSettings));
            return CmdParser.CombineSettings(settings) + " " + CmdParser.CombineSettings(SplitOnSemis(extraSettings));
        }

        private static char[] _dontSplitChars = new char[] { ' ', '=', '{', '}', '\t' };

        // REVIEW: Deprecate this!
        public static string[] SplitOnSemis(string[] args)
        {
            if (Utils.Size(args) == 0)
                return null;

            List<string> res = null;
            for (int i = 0; i < args.Length; i++)
            {
                string arg = args[i];

                if (!arg.Contains(';') || arg.IndexOfAny(_dontSplitChars) >= 0)
                {
                    if (res == null)
                        continue;
                    res.Add(arg);
                }
                else
                {
                    if (res == null)
                        res = new List<string>(args.Take(i));
                    res.AddRange(arg.Split(';'));
                }
            }

            return res == null ? args : res.ToArray();
        }

        /// <summary>
        /// Make a string representation of an array
        /// </summary>
        public static string Array2String(Float[] a, string sep)
        {
            StringBuilder sb = new StringBuilder();
            if (a.Length == 0)
                return "";
            sb.Append(a[0].ToString());
            for (int i = 1; i < a.Length; i++)
                sb.Append(sep + a[i]);
            return sb.ToString();
        }

        /// <summary>
        /// Convert string representation of char separator(s)
        /// </summary>
        public static char[] SeparatorFromString(string sep)
        {
            if (string.IsNullOrEmpty(sep))
                return null;
            if (sep.Length == 1)
                return new char[] { sep[0] };

            List<char> sepChars = new List<char>();
            foreach (string s in sep.Split(','))
            {
                char c = SepCharFromString(s);
                if (c != 0)
                    sepChars.Add(c);
            }
            return sepChars.Count > 0 ? sepChars.ToArray() : null;
        }

        /// <summary>
        /// Convert from a string representation of separator to a char
        /// </summary>
        public static char SepCharFromString(string s)
        {
            if (string.IsNullOrEmpty(s))
                return default(char);

            switch (s.ToLower())
            {
            case "space":
                return ' ';
            case "tab":
                return '\t';
            case "comma":
                return ',';
            case "colon":
                return ':';
            case "semicolon":
                return ';';
            case "bar": // VW format
                return '|';
            default:
                // REVIEW: This is bad - why do we simply ignore unexpected values?
                if (s.Length == 1)
                    return s[0];
                return default(char);
            }
        }
    }

    /// <summary>
    /// A generic reverse Comparer (for use in Array.Sort)
    /// </summary>
    public sealed class ReverseComparer<T> : IComparer<T>
        where T : IComparable<T>
    {
        public int Compare(T x, T y)
        {
            return -x.CompareTo(y);
        }
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.IO;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    // REVIEW: Implement properly on CoreCLR.
    public static class StreamUtils
    {
        public static Stream OpenInStream(string fileName)
        {
#if !CORECLR
            return Microsoft.ML.Runtime.Internal.IO.ZStreamIn.Open(fileName);
#else
            return new FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.Read);
#endif
        }

        public static StreamWriter CreateWriter(string fileName)
        {
#if !CORECLR
            return Internal.IO.ZStreamWriter.Open(fileName);
#else
            // REVIEW: Should specify the encoding.
            return new StreamWriter(new FileStream(fileName, FileMode.Create, FileAccess.Write));
#endif
        }

        public static string[] ExpandWildCards(string pattern)
        {
#if !CORECLR
            return Internal.IO.IOUtil.ExpandWildcards(pattern);
#else
            return Expand(pattern);
#endif
        }

#if CORECLR
        private static readonly char[] _wildChars = new char[] { '*', '?' };
        private static readonly char[] _wildPlusChars = new char[] { '*', '?', '+' };

        /// <summary>
        /// Expand an extended wildcard pattern into a set of file paths.
        /// </summary>
        /// <param name="pattern">the pattern to expand</param>
        /// <returns>the set of file paths matching the pattern</returns>
        /// <remarks>
        /// The wildcard pattern accepts the standard "*" and "?" placeholders.
        /// "..." also refers to a recursive search over subdirectories.
        /// "+" can also be used to make a union of several filenames or patterns.
        /// Names of files that do not exist will be excluded.
        /// </remarks>
        private static string[] Expand(string pattern)
        {
            if (pattern == null || (pattern.IndexOfAny(_wildPlusChars) < 0 && pattern.IndexOf("...") < 0))
            {
                if (File.Exists(pattern))
                    return new string[] { pattern };
                else
                    return new string[0];
            }
            var patterns = pattern.Split('+');
            var matchList = new List<string>();
            foreach (var currentPattern in patterns)
            {
                // hard-code in special types??
                if (currentPattern.Length == 0)
                    continue;
                var patLower = currentPattern.ToLower();
                if (currentPattern.IndexOfAny(_wildChars) >= 0 || currentPattern.IndexOf("...") >= 0)
                {
                    // compressed extensions are not automatically used! ***
                    int recursiveIndex = currentPattern.IndexOf("...");
                    if (recursiveIndex >= 0)
                    {
                        var left = currentPattern.Substring(0, recursiveIndex);
                        var right = currentPattern.Substring(recursiveIndex + 3);
                        right = right.TrimStart('\\', '/');
                        if (right.Length == 0)
                            right = "*";
                        var path = left;
                        var pathEmpty = (path == null || path.Length == 0);
                        if (pathEmpty)
                            path = ".";
                        var dirsLeft = new Stack<string>();
                        dirsLeft.Push(path);
                        while (dirsLeft.Count != 0)
                        {
                            var dir = dirsLeft.Pop();
                            // watch for lack of access:
                            try
                            {
                                // this is actually incorrect, for 3-char extensions: ***
                                var files = Directory.GetFiles(dir, right);
                                if (pathEmpty)
                                {
                                    for (int i = 0; i < files.Length; i++)
                                    {
                                        if (files[i].StartsWith("./") || files[i].StartsWith(".\\"))
                                            files[i] = files[i].Substring(2);
                                    }
                                }
                                matchList.AddRange(files);
                                var subs = Directory.GetDirectories(dir);
                                for (var i = subs.Length - 1; i >= 0; i--)
                                    dirsLeft.Push(subs[i]);
                            }
                            catch
                            {
                                // ignore
                            }
                        }
                    }
                    else
                    {
                        try
                        {
                            var path = Path.GetDirectoryName(currentPattern);
                            var pathEmpty = !(currentPattern.StartsWith("./") || currentPattern.StartsWith(".\\"));
                            if (path == null || path.Length == 0)
                                path = ".";
                            // watch for lack of access:
                            try
                            {
                                var files = Directory.GetFiles(path, Path.GetFileName(currentPattern));
                                if (pathEmpty)
                                {
                                    for (int i = 0; i < files.Length; i++)
                                    {
                                        if (files[i].StartsWith("./") || files[i].StartsWith(".\\"))
                                            files[i] = files[i].Substring(2);
                                    }
                                }
                                matchList.AddRange(files);
                            }
                            catch
                            {
                                // ignore
                            }
                        }
                        catch
                        {
                            // ignore bad path?
                        }
                    }
                }
                else
                {
                    // what to do?? Filter to only those that exist?? ***
                    if (!File.Exists(currentPattern))
                        continue;
                    matchList.Add(currentPattern);
                }
            }

            // remove duplicates, very inefficiently - but it is simple, preserves
            // the order, uses no additional memory, and is case-insensitive...:
            for (var i = 0; i < matchList.Count - 1; i++)
            {
                for (var j = i + 1; j < matchList.Count; j++)
                {
                    if (string.Compare(matchList[i], matchList[j], true) == 0)
                    {
                        matchList.RemoveAt(j);
                        j--;
                    }
                }
            }
            return matchList.ToArray();
        }
#endif
    }
}

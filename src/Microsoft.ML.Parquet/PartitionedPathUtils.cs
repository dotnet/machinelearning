﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;

namespace Microsoft.ML.Data.Utilities
{
    internal static class PartitionedPathUtils
    {
        /// <summary>
        /// Make a full path realtive to a base path.
        /// </summary>
        /// <param name="basepath">The base path, assumed to be a directory.</param>
        /// <param name="path">The full path.</param>
        /// <returns>The relative path.</returns>
        /// <exception cref="InvalidOperationException">If the paths are not relative.</exception>
        internal static string MakePathRelative(string basepath, string path)
        {
            Contracts.AssertNonEmpty(basepath);
            Contracts.AssertNonEmpty(path);

            Uri baseUri = new Uri(basepath);
            Uri uri = new Uri(path);

            if (baseUri.Scheme != uri.Scheme)
            {
                throw Contracts.ExceptParam(nameof(basepath), "Paths cannot be made relative as they are of different schemes.");
            }

            string relativePath;
            try
            {
                if (!baseUri.AbsoluteUri.EndsWith("/"))
                {
                    baseUri = new Uri(baseUri.AbsoluteUri + "/");
                }

                relativePath = Uri.UnescapeDataString(baseUri.MakeRelativeUri(uri).ToString());
            }
            catch (ArgumentNullException e)
            {
                throw Contracts.Except(e, "Paths could not be made relative.");
            }
            catch (InvalidOperationException e)
            {
                throw Contracts.Except(e, "Paths could not be made relative.");
            }

            if (uri.Scheme.Equals("file", StringComparison.OrdinalIgnoreCase))
            {
                relativePath = relativePath.Replace(Path.AltDirectorySeparatorChar, Path.DirectorySeparatorChar);
            }

            return relativePath;
        }

        /// <summary>
        /// Split a path string into an enumerable list of the directories.
        /// </summary>
        /// <param name="path">The path string to split.</param>
        /// <returns>An enumerable list of all non-empty directories.</returns>
        internal static IEnumerable<string> SplitDirectories(string path)
        {
            char[] separators = { Path.DirectorySeparatorChar };

            var cleanPath = path.Replace(Path.AltDirectorySeparatorChar, Path.DirectorySeparatorChar);
            return cleanPath.Split(separators, StringSplitOptions.RemoveEmptyEntries);
        }
    }
}

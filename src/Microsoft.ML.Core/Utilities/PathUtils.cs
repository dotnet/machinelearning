// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Threading;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    internal static partial class Utils
    {
        /// <summary>
        /// Environment variable containing optional resources path.
        /// </summary>
        public const string CustomSearchDirEnvVariable = "MICROSOFTML_RESOURCE_PATH";

        private static string _dllDir;

        private static string DllDir
        {
            get
            {
                if (_dllDir == null)
                {
                    string path = typeof(Utils).Assembly.Location;
                    string directory = Path.GetDirectoryName(path);
                    Interlocked.CompareExchange(ref _dllDir, directory, null);
                }

                return _dllDir;
            }
        }

        /// <summary>
        /// Attempts to find a file that is expected to be distributed with a TLC component. Searches
        /// in the following order:
        /// 1. In the customSearchDir directory, if it is provided.
        /// 2. In the custom search directory specified by the
        ///    <seealso cref="CustomSearchDirEnvVariable"/> environment variable.
        /// 3. In the root folder of the provided assembly.
        /// 4. In the folder of this assembly.
        /// In each case it searches the file in the directory provided and combined with folderPrefix.
        ///
        /// If any of these locations contain the file, a full local path will be returned, otherwise this
        /// method will return null.
        /// </summary>
        /// <param name="fileName">File name to find</param>
        /// <param name="folderPrefix">folder prefix, relative to the current or customSearchDir</param>
        /// <param name="customSearchDir">
        /// Custom directory to search for resources.
        /// If null, the path specified in the environment variable <seealso cref="CustomSearchDirEnvVariable"/>
        /// will be used.
        /// </param>
        /// <param name="assemblyForBasePath">
        /// Assembly type to search the path of.
        /// </param>
        /// <returns>
        /// Path to the existing file. Null if not found.
        /// </returns>
        public static string FindExistentFileOrNull(string fileName, string folderPrefix = null, string customSearchDir = null, System.Type assemblyForBasePath = null)
        {
            Contracts.AssertNonWhiteSpace(fileName);

            string candidate;

            // 1. Search in customSearchDir.
            if (!string.IsNullOrWhiteSpace(customSearchDir)
                && TryFindFile(fileName, folderPrefix, customSearchDir, out candidate))
            {
                return candidate;
            }

            // 2. Search in the path specified by the environment variable.
            var envDir = Environment.GetEnvironmentVariable(CustomSearchDirEnvVariable);
            if (!string.IsNullOrWhiteSpace(envDir)
                && TryFindFile(fileName, folderPrefix, envDir, out candidate))
            {
                return candidate;
            }

            // 3. Search in the path specified by the assemblyForBasePath.
            if (assemblyForBasePath != null)
            {
                // For CoreTLC we have MLCore located in main folder, and dlls with learners and transforms located
                // in AutoLoad folder or (ClrWin|ClrLinux) folder so we need to check folder for provided type.
                var assemblyDir = Path.GetDirectoryName(assemblyForBasePath.Assembly.Location);
                if (assemblyDir != null && customSearchDir != null)
                    assemblyDir = Path.Combine(assemblyDir, customSearchDir);
                if (TryFindFile(fileName, folderPrefix, assemblyDir, out candidate))
                    return candidate;
            }

            // 4. Fallback to the root path of the current assembly
            TryFindFile(fileName, folderPrefix, DllDir, out candidate);
            return candidate;
        }

        private static bool TryFindFile(string fileName, string folderPrefix, string dir, out string foundFile)
        {
            foundFile = null;
            var candidate = Path.Combine(dir, fileName);
            if (File.Exists(candidate))
            {
                foundFile = candidate;
                return true;
            }

            if (!string.IsNullOrWhiteSpace(folderPrefix))
            {
                candidate = Path.Combine(dir, folderPrefix, fileName);
                if (File.Exists(candidate))
                {
                    foundFile = candidate;
                    return true;
                }
            }

            return false;
        }

        /// <summary>
        ///  Given a folder path, create it if it doesn't exist.
        ///  Fails if the folder name is empty, or can't create the folder.
        /// </summary>
        public static string CreateFolderIfNotExists(string folder)
        {
            if (Directory.Exists(folder))
                return folder;

            if (!string.IsNullOrEmpty(folder))
            {
                try
                {
                    Directory.CreateDirectory(folder);
                    return folder;
                }
                catch (Exception exc)
                {
                    throw Contracts.ExceptParam(nameof(folder), $"Failed to create folder for the provided path: {folder}. \nException: {exc.Message}");
                }
            }

            return null;
        }
    }
}

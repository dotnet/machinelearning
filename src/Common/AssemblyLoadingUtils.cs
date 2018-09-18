// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Internal.Utilities;
using System;
using System.IO;
using System.IO.Compression;
using System.Reflection;

namespace Microsoft.ML.Runtime
{
    internal static class AssemblyLoadingUtils
    {
        /// <summary>
        /// Make sure the given assemblies are loaded and that their loadable classes have been catalogued.
        /// </summary>
        public static void LoadAndRegister(IHostEnvironment env, string[] assemblies)
        {
            Contracts.AssertValue(env);

            if (Utils.Size(assemblies) > 0)
            {
                foreach (string path in assemblies)
                {
                    Exception ex = null;
                    try
                    {
                        // REVIEW: Will LoadFrom ever return null?
                        Contracts.CheckNonEmpty(path, nameof(path));
                        var assem = LoadAssembly(env, path);
                        if (assem != null)
                            continue;
                    }
                    catch (Exception e)
                    {
                        ex = e;
                    }

                    // If it is a zip file, load it that way.
                    ZipArchive zip;
                    try
                    {
                        zip = ZipFile.OpenRead(path);
                    }
                    catch (Exception e)
                    {
                        // Couldn't load as an assembly and not a zip, so warn the user.
                        ex = ex ?? e;
                        Console.Error.WriteLine("Warning: Could not load '{0}': {1}", path, ex.Message);
                        continue;
                    }

                    string dir;
                    try
                    {
                        dir = CreateTempDirectory();
                    }
                    catch (Exception e)
                    {
                        throw Contracts.ExceptIO(e, "Creating temp directory for extra assembly zip extraction failed: '{0}'", path);
                    }

                    try
                    {
                        zip.ExtractToDirectory(dir);
                    }
                    catch (Exception e)
                    {
                        throw Contracts.ExceptIO(e, "Extracting extra assembly zip failed: '{0}'", path);
                    }

                    LoadAssembliesInDir(env, dir);
                }
            }
        }

        private static string CreateTempDirectory()
        {
            string dir = GetTempPath();
            Directory.CreateDirectory(dir);
            return dir;
        }

        private static string GetTempPath()
        {
            Guid guid = Guid.NewGuid();
            return Path.GetFullPath(Path.Combine(Path.GetTempPath(), "MLNET_" + guid.ToString()));
        }

        private static void LoadAssembliesInDir(IHostEnvironment env, string dir)
        {
            if (!Directory.Exists(dir))
                return;

            // Load all dlls in the given directory.
            var paths = Directory.EnumerateFiles(dir, "*.dll");
            foreach (string path in paths)
            {
                LoadAssembly(env, path);
            }
        }

        /// <summary>
        /// Given an assembly path, load the assembly and register it with the ComponentCatalog.
        /// </summary>
        private static Assembly LoadAssembly(IHostEnvironment env, string path)
        {
            try
            {
                var assembly = Assembly.LoadFrom(path);
                env.ComponentCatalog.RegisterAssembly(assembly);
                return assembly;
            }
            catch (Exception)
            {
                return null;
            }
        }
    }
}

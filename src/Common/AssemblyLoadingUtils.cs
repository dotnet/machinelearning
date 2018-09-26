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

                    LoadAssembliesInDir(env, dir, false);
                }
            }
        }

        public static IDisposable CreateAssemblyRegistrar(IHostEnvironment env, string loadAssembliesPath = null)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValueOrNull(loadAssembliesPath);

            return new AssemblyRegistrar(env, loadAssembliesPath);
        }

        public static void RegisterCurrentLoadedAssemblies(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));

            foreach (Assembly a in AppDomain.CurrentDomain.GetAssemblies())
            {
                TryRegisterAssembly(env.ComponentCatalog, a);
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

        private static readonly string[] _filePrefixesToAvoid = new string[] {
            "api-ms-win",
            "clr",
            "coreclr",
            "dbgshim",
            "ext-ms-win",
            "microsoft.bond.",
            "microsoft.cosmos.",
            "microsoft.csharp",
            "microsoft.data.",
            "microsoft.hpc.",
            "microsoft.live.",
            "microsoft.platformbuilder.",
            "microsoft.visualbasic",
            "microsoft.visualstudio.",
            "microsoft.win32",
            "microsoft.windowsapicodepack.",
            "microsoft.windowsazure.",
            "mscor",
            "msvc",
            "petzold.",
            "roslyn.",
            "sho",
            "sni",
            "sqm",
            "system.",
            "zlib",
        };

        private static bool ShouldSkipPath(string path)
        {
            string name = Path.GetFileName(path).ToLowerInvariant();
            switch (name)
            {
                case "cpumathnative.dll":
                case "cqo.dll":
                case "fasttreenative.dll":
                case "factorizationmachinenative.dll":
                case "libiomp5md.dll":
                case "ldanative.dll":
                case "libvw.dll":
                case "matrixinterf.dll":
                case "microsoft.ml.neuralnetworks.gpucuda.dll":
                case "mklimports.dll":
                case "microsoft.research.controls.decisiontrees.dll":
                case "microsoft.ml.neuralnetworks.sse.dll":
                case "neuraltreeevaluator.dll":
                case "optimizationbuilderdotnet.dll":
                case "parallelcommunicator.dll":
                case "microsoft.ml.runtime.runtests.dll":
                case "scopecompiler.dll":
                case "tbb.dll":
                case "internallearnscope.dll":
                case "unmanagedlib.dll":
                case "vcclient.dll":
                case "libxgboost.dll":
                case "zedgraph.dll":
                case "__scopecodegen__.dll":
                case "cosmosClientApi.dll":
                    return true;
            }

            foreach (var s in _filePrefixesToAvoid)
            {
                if (name.StartsWith(s, StringComparison.OrdinalIgnoreCase))
                    return true;
            }

            return false;
        }

        private static void LoadAssembliesInDir(IHostEnvironment env, string dir, bool filter)
        {
            if (!Directory.Exists(dir))
                return;

            // Load all dlls in the given directory.
            var paths = Directory.EnumerateFiles(dir, "*.dll");
            foreach (string path in paths)
            {
                if (filter && ShouldSkipPath(path))
                {
                    continue;
                }

                LoadAssembly(env, path);
            }
        }

        /// <summary>
        /// Given an assembly path, load the assembly and register it with the ComponentCatalog.
        /// </summary>
        private static Assembly LoadAssembly(IHostEnvironment env, string path)
        {
            Assembly assembly = null;
            try
            {
                assembly = Assembly.LoadFrom(path);
            }
            catch (Exception)
            {
                return null;
            }

            if (assembly != null)
            {
                TryRegisterAssembly(env.ComponentCatalog, assembly);
            }

            return assembly;
        }

        /// <summary>
        /// Checks whether <paramref name="assembly"/> references the assembly containing LoadableClassAttributeBase,
        /// and therefore can contain components.
        /// </summary>
        private static bool CanContainComponents(Assembly assembly)
        {
            var targetFullName = typeof(LoadableClassAttributeBase).Assembly.GetName().FullName;

            bool found = false;
            foreach (var name in assembly.GetReferencedAssemblies())
            {
                if (name.FullName == targetFullName)
                {
                    found = true;
                    break;
                }
            }

            return found;
        }

        private static void TryRegisterAssembly(ComponentCatalog catalog, Assembly assembly)
        {
            // Don't try to index dynamic generated assembly
            if (assembly.IsDynamic)
                return;

            if (!CanContainComponents(assembly))
                return;

            catalog.RegisterAssembly(assembly);
        }

        private sealed class AssemblyRegistrar : IDisposable
        {
            private readonly IHostEnvironment _env;

            public AssemblyRegistrar(IHostEnvironment env, string path)
            {
                _env = env;

                RegisterCurrentLoadedAssemblies(_env);

                if (!string.IsNullOrEmpty(path))
                {
                    LoadAssembliesInDir(_env, path, true);
                    path = Path.Combine(path, "AutoLoad");
                    LoadAssembliesInDir(_env, path, true);
                }

                AppDomain.CurrentDomain.AssemblyLoad += CurrentDomainAssemblyLoad;
            }

            public void Dispose()
            {
                AppDomain.CurrentDomain.AssemblyLoad -= CurrentDomainAssemblyLoad;
            }

            private void CurrentDomainAssemblyLoad(object sender, AssemblyLoadEventArgs args)
            {
                TryRegisterAssembly(_env.ComponentCatalog, args.LoadedAssembly);
            }
        }
    }
}

// Taken from https://raw.githubusercontent.com/mellinoe/nativelibraryloader/586f9738ff12688df8f0662027da8c319aee3841/NativeLibraryLoader/PathResolver.cs
using Microsoft.DotNet.PlatformAbstractions;
using Microsoft.Extensions.DependencyModel;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

namespace Microsoft.ML.TestFrameworkCommon.Utility
{
    /// <summary>
    /// Enumerates possible library load targets.
    /// </summary>
    public abstract class PathResolver
    {
        /// <summary>
        /// Returns an enumerator which yields possible library load targets, in priority order.
        /// </summary>
        /// <param name="name">The name of the library to load.</param>
        /// <returns>An enumerator yielding load targets.</returns>
        public abstract IEnumerable<string> EnumeratePossibleLibraryLoadTargets(string name);

        /// <summary>
        /// Gets a default path resolver.
        /// </summary>
        public static PathResolver Default { get; } = new DefaultPathResolver();
    }

    /// <summary>
    /// Enumerates possible library load targets. This default implementation returns the following load targets:
    /// First: The library contained in the applications base folder.
    /// Second: The simple name, unchanged.
    /// Third: The library as resolved via the default DependencyContext, in the default nuget package cache folder.
    /// </summary>
    public class DefaultPathResolver : PathResolver
    {
        /// <summary>
        /// Returns an enumerator which yields possible library load targets, in priority order.
        /// </summary>
        /// <param name="name">The name of the library to load.</param>
        /// <returns>An enumerator yielding load targets.</returns>
        public override IEnumerable<string> EnumeratePossibleLibraryLoadTargets(string name)
        {
            if (!string.IsNullOrEmpty(AppContext.BaseDirectory))
            {
                yield return Path.Combine(AppContext.BaseDirectory, name);
            }
            yield return name;
            if (TryLocateNativeAssetFromDeps(name, out string appLocalNativePath, out string depsResolvedPath))
            {
                yield return appLocalNativePath;
                yield return depsResolvedPath;
            }
        }

        private bool TryLocateNativeAssetFromDeps(string name, out string appLocalNativePath, out string depsResolvedPath)
        {
            DependencyContext defaultContext = DependencyContext.Default;
            if (defaultContext == null)
            {
                appLocalNativePath = null;
                depsResolvedPath = null;
                return false;
            }

#pragma warning disable MSML_ParameterLocalVarName // Parameter or local variable name not standard
            string currentRID = DotNet.PlatformAbstractions.RuntimeEnvironment.GetRuntimeIdentifier();
#pragma warning restore MSML_ParameterLocalVarName // Parameter or local variable name not standard

            List<string> allRIDs = new List<string>();
            allRIDs.Add(currentRID);
            if (!AddFallbacks(allRIDs, currentRID, defaultContext.RuntimeGraph))
            {
#pragma warning disable MSML_ParameterLocalVarName // Parameter or local variable name not standard
                string guessedFallbackRID = GuessFallbackRID(currentRID);
#pragma warning restore MSML_ParameterLocalVarName // Parameter or local variable name not standard

                if (guessedFallbackRID != null)
                {
                    allRIDs.Add(guessedFallbackRID);
                    AddFallbacks(allRIDs, guessedFallbackRID, defaultContext.RuntimeGraph);
                }
            }

            foreach (string rid in allRIDs)
            {
                foreach (var runtimeLib in defaultContext.RuntimeLibraries)
                {
                    foreach (var nativeAsset in runtimeLib.GetRuntimeNativeAssets(defaultContext, rid))
                    {
                        if (Path.GetFileName(nativeAsset) == name || Path.GetFileNameWithoutExtension(nativeAsset) == name)
                        {
                            appLocalNativePath = Path.Combine(
                                AppContext.BaseDirectory,
                                nativeAsset);
                            appLocalNativePath = Path.GetFullPath(appLocalNativePath);

                            depsResolvedPath = Path.Combine(
                                GetNugetPackagesRootDirectory(),
                                runtimeLib.Name.ToLowerInvariant(),
                                runtimeLib.Version,
                                nativeAsset);
                            depsResolvedPath = Path.GetFullPath(depsResolvedPath);

                            return true;
                        }
                    }
                }
            }

            appLocalNativePath = null;
            depsResolvedPath = null;
            return false;
        }

        private string GuessFallbackRID(string actualRuntimeIdentifier)
        {
            if (actualRuntimeIdentifier == "osx.10.13-x64")
            {
                return "osx.10.12-x64";
            }
            else if (actualRuntimeIdentifier.StartsWith("osx"))
            {
                return "osx-x64";
            }

            return null;
        }

        private bool AddFallbacks(List<string> fallbacks, string rid, IReadOnlyList<RuntimeFallbacks> allFallbacks)
        {
            foreach (RuntimeFallbacks fb in allFallbacks)
            {
                if (fb.Runtime == rid)
                {
                    fallbacks.AddRange(fb.Fallbacks);
                    return true;
                }
            }

            return false;
        }

        private string GetNugetPackagesRootDirectory()
        {
            // TODO: Handle alternative package directories, if they are configured.
            return Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nuget", "packages");
        }
    }
}

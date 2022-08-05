// Taken from https://raw.githubusercontent.com/mellinoe/nativelibraryloader/586f9738ff12688df8f0662027da8c319aee3841/NativeLibraryLoader/PathResolver.cs
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
#if NETCOREAPP
using System.Runtime.Loader;
#endif

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
    /// Third: On .NETCore the library as resolved via AssemblyDependencyResolver, which uses information from the 
    /// AssemblyLoadContext / application deps file to locate the assembly in either an application subfolder or the
    /// NuGet packages folder.
    /// </summary>
    public class DefaultPathResolver : PathResolver
    {
#if NETCOREAPP
        private AssemblyDependencyResolver _dependencyResolver = null;

        public DefaultPathResolver()
        {
            Assembly entryAssembly = Assembly.GetEntryAssembly();

            if (entryAssembly != null && entryAssembly.Location != null)
            {
                _dependencyResolver = new AssemblyDependencyResolver(entryAssembly.Location);
            }
        }
#endif

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

#if NETCOREAPP
            if (_dependencyResolver != null)
            {
                yield return _dependencyResolver.ResolveUnmanagedDllToPath(name);
            }
#endif
        }
    }
}

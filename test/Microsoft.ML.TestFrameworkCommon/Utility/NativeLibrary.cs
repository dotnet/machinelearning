// Taken from https://github.com/mellinoe/nativelibraryloader/blob/586f9738ff12688df8f0662027da8c319aee3841/NativeLibraryLoader/NativeLibrary.cs
using System;
using System.Runtime.InteropServices;
#if NET5_0_OR_GREATER
using RuntimeNativeLibrary = System.Runtime.InteropServices.NativeLibrary;
#endif

namespace Microsoft.ML.TestFrameworkCommon.Utility
{
    /// <summary>
    /// Represents a native shared library opened by the operating system.
    /// </summary>
    public class NativeLibrary : IDisposable
    {
        private static readonly LibraryLoader _platformDefaultLoader = LibraryLoader.GetPlatformDefaultLoader();
        private readonly LibraryLoader _loader;

        public static bool NativeLibraryExists(string name)
        {
#if NET5_0_OR_GREATER
            // Use the runtime's assembly-aware loader to honor native NuGet asset probing.
            // The legacy loader below can falsely report missing libraries on Linux: its "libdl"
            // import may not resolve to libdl.so.2, and its distro-specific RID lookup can miss
            // portable linux-x64 assets. Passing the owning assembly avoids that custom probing.
            var assembly = typeof(NativeLibrary).Assembly;
            if (!RuntimeNativeLibrary.TryLoad(name, assembly, null, out var handle) &&
                !RuntimeNativeLibrary.TryLoad("lib" + name, assembly, null, out handle) &&
                !RuntimeNativeLibrary.TryLoad("lib_" + name, assembly, null, out handle))
            {
                return false;
            }

            RuntimeNativeLibrary.Free(handle);
            return true;
#else
            NativeLibrary nativeLibrary = default;
            string extension = default;
            string prefix = "lib";

            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                extension = ".dylib";
            else if (Environment.OSVersion.Platform == PlatformID.Unix)
                extension = ".so";
            else
                extension = ".dll";

            try
            {
                // Try the native name as is.
                nativeLibrary = new NativeLibrary(name + extension);
                return true;
            }
            catch
            {
                // If that didn't load, dispose of the first attempt and try appending lib in front
                try
                {
                    nativeLibrary?.Dispose();
                    nativeLibrary = new NativeLibrary(prefix + name + extension);
                    return true;
                }
                catch
                {
                    // If that didn't load, dispose of the first attempt and try appending lib_ in front
                    try
                    {
                        nativeLibrary?.Dispose();
                        nativeLibrary = new NativeLibrary(prefix + "_" + name + extension);
                        return true;
                    }
                    catch
                    {
                        return false;
                    }
                }
            }
            finally
            {
                nativeLibrary?.Dispose();
            }
#endif
        }

        /// <summary>
        /// The operating system handle of the loaded library.
        /// </summary>
        private IntPtr Handle { get; }

        /// <summary>
        /// Constructs a new NativeLibrary using the platform's default library loader.
        /// </summary>
        /// <param name="name">The name of the library to load.</param>
        private NativeLibrary(string name)
        {
            _loader = _platformDefaultLoader;
            Handle = _loader.LoadNativeLibrary(name, PathResolver.Default);
        }

        /// <summary>
        /// Frees the native library.
        /// </summary>
        public void Dispose()
        {
            _loader.FreeNativeLibrary(Handle);
        }
    }
}

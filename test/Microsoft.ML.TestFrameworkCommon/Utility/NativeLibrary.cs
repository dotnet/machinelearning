// Taken from https://github.com/mellinoe/nativelibraryloader/blob/586f9738ff12688df8f0662027da8c319aee3841/NativeLibraryLoader/NativeLibrary.cs
using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.TestFrameworkCommon.Utility
{
    /// <summary>
    /// Represents a native shared library opened by the operating system.
    /// This type can be used to load native function pointers by name.
    /// </summary>
    public class NativeLibrary : IDisposable
    {
        private static readonly LibraryLoader _platformDefaultLoader = LibraryLoader.GetPlatformDefaultLoader();
        private readonly LibraryLoader _loader;

        public static bool NativeLibraryExists(string name)
        {
            NativeLibrary nativeLibrary = default;
            string extension = default;
            string prefix = "lib";

            if (Environment.OSVersion.Platform == PlatformID.MacOSX)
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
                    return false;
                }
            }
            finally
            {
                nativeLibrary?.Dispose();
            }
        }

        /// <summary>
        /// The operating system handle of the loaded library.
        /// </summary>
        public IntPtr Handle { get; }

        /// <summary>
        /// Constructs a new NativeLibrary using the platform's default library loader.
        /// </summary>
        /// <param name="name">The name of the library to load.</param>
        public NativeLibrary(string name) : this(name, _platformDefaultLoader, PathResolver.Default)
        {
        }

        /// <summary>
        /// Constructs a new NativeLibrary using the platform's default library loader.
        /// </summary>
        /// <param name="names">An ordered list of names to attempt to load.</param>
        public NativeLibrary(string[] names) : this(names, _platformDefaultLoader, PathResolver.Default)
        {
        }

        /// <summary>
        /// Constructs a new NativeLibrary using the specified library loader.
        /// </summary>
        /// <param name="name">The name of the library to load.</param>
        /// <param name="loader">The loader used to open and close the library, and to load function pointers.</param>
        public NativeLibrary(string name, LibraryLoader loader) : this(name, loader, PathResolver.Default)
        {
        }

        /// <summary>
        /// Constructs a new NativeLibrary using the specified library loader.
        /// </summary>
        /// <param name="names">An ordered list of names to attempt to load.</param>
        /// <param name="loader">The loader used to open and close the library, and to load function pointers.</param>
        public NativeLibrary(string[] names, LibraryLoader loader) : this(names, loader, PathResolver.Default)
        {
        }

        /// <summary>
        /// Constructs a new NativeLibrary using the specified library loader.
        /// </summary>
        /// <param name="name">The name of the library to load.</param>
        /// <param name="loader">The loader used to open and close the library, and to load function pointers.</param>
        /// <param name="pathResolver">The path resolver, used to identify possible load targets for the library.</param>
        public NativeLibrary(string name, LibraryLoader loader, PathResolver pathResolver)
        {
            _loader = loader;
            Handle = _loader.LoadNativeLibrary(name, pathResolver);
        }

        /// <summary>
        /// Constructs a new NativeLibrary using the specified library loader.
        /// </summary>
        /// <param name="names">An ordered list of names to attempt to load.</param>
        /// <param name="loader">The loader used to open and close the library, and to load function pointers.</param>
        /// <param name="pathResolver">The path resolver, used to identify possible load targets for the library.</param>
        public NativeLibrary(string[] names, LibraryLoader loader, PathResolver pathResolver)
        {
            _loader = loader;
            Handle = _loader.LoadNativeLibrary(names, pathResolver);
        }

        /// <summary>
        /// Loads a function whose signature matches the given delegate type's signature.
        /// </summary>
        /// <typeparam name="T">The type of delegate to return.</typeparam>
        /// <param name="name">The name of the native export.</param>
        /// <returns>A delegate wrapping the native function.</returns>
        /// <exception cref="InvalidOperationException">Thrown when no function with the given name
        /// is exported from the native library.</exception>
        public T LoadFunction<T>(string name)
        {
            IntPtr functionPtr = _loader.LoadFunctionPointer(Handle, name);
            if (functionPtr == IntPtr.Zero)
            {
                throw new InvalidOperationException($"No function was found with the name {name}.");
            }

            return Marshal.GetDelegateForFunctionPointer<T>(functionPtr);
        }

        /// <summary>
        /// Loads a function pointer with the given name.
        /// </summary>
        /// <param name="name">The name of the native export.</param>
        /// <returns>A function pointer for the given name, or 0 if no function with that name exists.</returns>
        public IntPtr LoadFunction(string name)
        {
            return _loader.LoadFunctionPointer(Handle, name);
        }

        /// <summary>
        /// Frees the native library. Function pointers retrieved from this library will be void.
        /// </summary>
        public void Dispose()
        {
            _loader.FreeNativeLibrary(Handle);
        }
    }
}

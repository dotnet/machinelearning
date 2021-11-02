// Taken from https://github.com/mellinoe/nativelibraryloader/blob/586f9738ff12688df8f0662027da8c319aee3841/NativeLibraryLoader/Libdl.cs
using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.TestFrameworkCommon.Utility
{
    internal static class Libdl
    {
        private const string LibName = "libdl";

        public const int RTLD_NOW = 0x002;

        [DllImport(LibName)]
        public static extern IntPtr dlopen(string fileName, int flags);

        [DllImport(LibName)]
        public static extern IntPtr dlsym(IntPtr handle, string name);

        [DllImport(LibName)]
        public static extern int dlclose(IntPtr handle);

        [DllImport(LibName)]
        public static extern string dlerror();
    }
}

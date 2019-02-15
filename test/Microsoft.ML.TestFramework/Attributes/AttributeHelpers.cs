// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Microsoft.ML.TestFramework.Attributes
{
    internal static class AttributeHelpers
    {
        internal static bool CheckLibcVersionGreaterThanMinimum(double minVersion)
        {
#if !NETFRAMEWORK
            string version;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return false;
            }

            try
            {
                version = Marshal.PtrToStringUTF8(gnu_get_libc_version());
            }
            catch (Exception e)
            {
                Debug.Assert(e is DllNotFoundException || e is EntryPointNotFoundException);
                return false;
            }

            return double.Parse(version) >= minVersion;
#else
            return false;
#endif
        }

        [DllImport("libc", ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr gnu_get_libc_version();
    }
}
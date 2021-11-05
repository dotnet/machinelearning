// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Microsoft.ML.TestFrameworkCommon.Attributes;

namespace Microsoft.ML.TestFramework.Attributes
{
    /// <summary>
    /// A fact for the Featurizers tests that wont run on CentOS7 and need the Featurizers library.
    /// </summary>
    public sealed class FeaturizersFactAttribute : EnvironmentSpecificFactAttribute
    {
        public FeaturizersFactAttribute() : base("These tests are not CentOS7 compliant and need the Featurizers native library.")
        {
        }
        protected override bool IsEnvironmentSupported()
        {
            // Featurizers.dll must exist
            if (!Microsoft.ML.TestFrameworkCommon.Utility.NativeLibrary.NativeLibraryExists("Featurizers"))
            {
                return false;
            }

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                using (Process process = new Process())
                {
                    process.StartInfo.FileName = "/bin/bash";
                    process.StartInfo.Arguments = "-c \"cat /etc/*-release\"";
                    process.StartInfo.UseShellExecute = false;
                    process.StartInfo.RedirectStandardOutput = true;
                    process.StartInfo.CreateNoWindow = true;
                    process.Start();

                    string distro = process.StandardOutput.ReadToEnd().Trim();

                    process.WaitForExit();
                    if (distro.Contains("CentOS Linux 7"))
                    {
                        return false;
                    }
                }
            }
            return true;
        }
    }
}

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
    /// A fact for tests that wont run on CentOS7
    /// </summary>
    public sealed class NotCentOS7FactAttribute : EnvironmentSpecificFactAttribute
    {
        public NotCentOS7FactAttribute() : base("These tests are not CentOS7 compliant.")
        {
        }
        protected override bool IsEnvironmentSupported()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                using (Process process = new Process())
                {
                    process.StartInfo.FileName = "/bin/bash";
                    process.StartInfo.Arguments= "-c \"cat /etc/*-release\"";
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
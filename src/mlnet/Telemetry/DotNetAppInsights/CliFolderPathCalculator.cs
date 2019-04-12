// Copyright (c) .NET Foundation and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.DotNet.Cli.Utils;

namespace Microsoft.DotNet.Configurer
{
    public static class CliFolderPathCalculator
    {
        public const string DotnetHomeVariableName = "DOTNET_CLI_HOME";
        private const string DotnetProfileDirectoryName = ".dotnet";
        private const string ToolsShimFolderName = "tools";
        private const string ToolsResolverCacheFolderName = "toolResolverCache";

        public static string CliFallbackFolderPath =>
            Environment.GetEnvironmentVariable("DOTNET_CLI_TEST_FALLBACKFOLDER") ??
                Path.Combine(new DirectoryInfo(AppContext.BaseDirectory).Parent.FullName, "NuGetFallbackFolder");

        public static string ToolsShimPath => Path.Combine(DotnetUserProfileFolderPath, ToolsShimFolderName);

        public static string ToolsPackagePath => ToolPackageFolderPathCalculator.GetToolPackageFolderPath(ToolsShimPath);

        public static BashPathUnderHomeDirectory ToolsShimPathInUnix =>
            new BashPathUnderHomeDirectory(
                DotnetHomePath,
                Path.Combine(DotnetProfileDirectoryName, ToolsShimFolderName));

        public static string DotnetUserProfileFolderPath =>
            Path.Combine(DotnetHomePath, DotnetProfileDirectoryName);

        public static string ToolsResolverCachePath => Path.Combine(DotnetUserProfileFolderPath, ToolsResolverCacheFolderName);

        public static string PlatformHomeVariableName =>
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "USERPROFILE" : "HOME";

        public static string DotnetHomePath
        {
            get
            {
                var home = Environment.GetEnvironmentVariable(DotnetHomeVariableName);
                if (string.IsNullOrEmpty(home))
                {
                    home = Environment.GetEnvironmentVariable(PlatformHomeVariableName);
                    if (string.IsNullOrEmpty(home))
                    {
                        throw new ConfigurationException(
                            string.Format(
                                "The user's home directory could not be determined. Set the '{0}' environment variable to specify the directory to use.",
                                DotnetHomeVariableName))
                            .DisplayAsError();
                    }
                }

                return home;
            }
        }
    }
}

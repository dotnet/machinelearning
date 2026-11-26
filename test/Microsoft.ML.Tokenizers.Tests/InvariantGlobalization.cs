// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;
using Microsoft.DotNet.RemoteExecutor;
using Xunit;

namespace Microsoft.ML.Tokenizers.Tests
{
    // Not in Utils.cs, which Microsoft.ML.Tokenizers.Data.Tests also compiles without referencing RemoteExecutor.
    internal static class InvariantGlobalization
    {
        // Runs the remote process in globalization-invariant mode. RuntimeConfigurationOptions would replace the test's
        // runtimeconfig.dev.json, which coverage builds need to find NuGet assemblies, so set the environment variable instead.
        public static RemoteInvokeOptions CreateRemoteInvokeOptions()
        {
            RemoteInvokeOptions options = new();
            options.StartInfo.Environment["DOTNET_SYSTEM_GLOBALIZATION_INVARIANT"] = "true";
            return options;
        }

        public static void AssertEnabled()
        {
            // string.Normalize returns its input unchanged when globalization-invariant mode is enabled.
            Assert.Equal("\u00E9", "\u00E9".Normalize(NormalizationForm.FormD));
        }
    }
}

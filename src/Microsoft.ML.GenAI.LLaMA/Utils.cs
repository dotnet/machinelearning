// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Reflection;
using TorchSharp;
using static TorchSharp.torch;

namespace Microsoft.ML.GenAI.LLaMA;

internal static class Utils
{
    public static string GetEmbeddedResource(string resourceName)
    {
        // read file content from embedded resource
        var assembly = Assembly.GetExecutingAssembly();
        var resourceStream = assembly.GetManifestResourceStream(resourceName);

        if (resourceStream == null)
        {
            throw new ArgumentException("Resource not found", resourceName);
        }

        using var reader = new System.IO.StreamReader(resourceStream);
        return reader.ReadToEnd();
    }
}

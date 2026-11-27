// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Concurrent;
using System.IO;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using Xunit;

namespace Microsoft.ML.RunTests;

[CollectionDefinition(nameof(AssemblyLoadingTestsCollection), DisableParallelization = true)]
public sealed class AssemblyLoadingTestsCollection
{
}

// BaseTestClass initializes native MKL in GlobalBase.AssemblyInit. This test only needs managed assembly loading.
#pragma warning disable MSML_ExtendBaseTestClass
[Collection(nameof(AssemblyLoadingTestsCollection))]
public class AssemblyLoadingUtilsTests
{
    [Theory]
    [InlineData("Microsoft.ML.runtests.dll")]
    [InlineData("cosmosClientApi.dll")]
    public void CreateAssemblyRegistrar_SkipsSpecifiedAssemblies(string fileName)
    {
        var environment = new MLContext();
        var sourcePath = Path.Combine(AppContext.BaseDirectory, fileName);
        Assert.True(File.Exists(sourcePath), $"Could not find fixture assembly '{sourcePath}'.");

        var sourceIdentity = AssemblyName.GetAssemblyName(sourcePath).Name;
        Assert.DoesNotContain(AppDomain.CurrentDomain.GetAssemblies(), assembly =>
            string.Equals(assembly.GetName().Name, sourceIdentity, StringComparison.OrdinalIgnoreCase));

        var temporaryDirectory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(temporaryDirectory);
        File.Copy(sourcePath, Path.Combine(temporaryDirectory, fileName));

        var loadedAssemblyNames = new ConcurrentBag<string>();
        void OnAssemblyLoad(object sender, AssemblyLoadEventArgs eventArgs)
        {
            loadedAssemblyNames.Add(eventArgs.LoadedAssembly.GetName().Name);
        }

        AppDomain.CurrentDomain.AssemblyLoad += OnAssemblyLoad;
        try
        {
#pragma warning disable CS0618
            using (AssemblyLoadingUtils.CreateAssemblyRegistrar(environment, temporaryDirectory))
            {
            }
#pragma warning restore CS0618

            Assert.DoesNotContain(sourceIdentity, loadedAssemblyNames, StringComparer.OrdinalIgnoreCase);
        }
        finally
        {
            AppDomain.CurrentDomain.AssemblyLoad -= OnAssemblyLoad;
            Directory.Delete(temporaryDirectory, recursive: true);
        }
    }
}
#pragma warning restore MSML_ExtendBaseTestClass

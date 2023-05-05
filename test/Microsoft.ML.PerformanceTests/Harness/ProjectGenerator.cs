// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using BenchmarkDotNet.Loggers;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Toolchains;
using BenchmarkDotNet.Toolchains.CsProj;

namespace Microsoft.ML.PerformanceTests.Harness
{
    /// <summary>
    /// to avoid side effects of benchmarks affect each other BenchmarkDotNet runs every benchmark in a standalone, dedicated process
    /// however to do that it needs to be able to create, build and run new executable
    /// 
    /// the problem with ML.NET is that it has native dependencies, which are NOT copied by MSBuild to the output folder
    /// in case where A has native dependency and B references A
    /// 
    /// this is why this class exists: 
    ///  1. to tell MSBuild to copy the native dependencies to folder with .exe (NativeAssemblyReference)
    ///  2. to generate a .csproj file that does not exclude Directory.Build.props (default BDN behavior) which contains custom NuGet feeds that are required for restore step
    /// </summary>
    public class ProjectGenerator : CsProjGenerator
    {
        private readonly string _runtimeIdentifier = string.Empty;

        public ProjectGenerator(string targetFrameworkMoniker) : base(targetFrameworkMoniker, null, null, null)
        {
#if NETFRAMEWORK
            _runtimeIdentifier = "win-x64";
#endif
        }

        protected override void GenerateProject(BuildPartition buildPartition, ArtifactsPaths artifactsPaths, ILogger logger)
            => File.WriteAllText(artifactsPaths.ProjectFilePath, $@"
<Project Sdk=""Microsoft.NET.Sdk"">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <OutputPath>bin\{buildPartition.BuildConfiguration}</OutputPath>
    <TargetFramework>{TargetFrameworkMoniker}</TargetFramework>
    <RuntimeIdentifier>{_runtimeIdentifier}</RuntimeIdentifier>
    <AssemblyName>{artifactsPaths.ProgramName}</AssemblyName>
    <AssemblyTitle>{artifactsPaths.ProgramName}</AssemblyTitle>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
    <TreatWarningsAsErrors>False</TreatWarningsAsErrors>
    <DebugType>pdbonly</DebugType>
    <DebugSymbols>true</DebugSymbols>
  </PropertyGroup>
  {GetRuntimeSettings(buildPartition.RepresentativeBenchmarkCase.Job.Environment.Gc, buildPartition.Resolver)}
  <ItemGroup>
    <Compile Include=""{Path.GetFileName(artifactsPaths.ProgramCodePath)}"" Exclude=""bin\**;obj\**;**\*.xproj;packages\**"" />
  </ItemGroup>
  <ItemGroup>
    <ProjectReference Include=""{GetProjectFilePath(buildPartition.RepresentativeBenchmarkCase.Descriptor.Type, logger).FullName}"" />
    {GenerateNativeReferences(buildPartition, logger)}
  </ItemGroup>
</Project>");

        // This overrides the .exe path to also involve the runtimeIdentifier for .NET Framework
        protected override string GetBinariesDirectoryPath(string buildArtifactsDirectoryPath, string configuration)
            => Path.Combine(buildArtifactsDirectoryPath, "bin", configuration, TargetFrameworkMoniker, _runtimeIdentifier);

        private string GenerateNativeReferences(BuildPartition buildPartition, ILogger logger)
        {
            var csproj = GetProjectFilePath(buildPartition.RepresentativeBenchmarkCase.Descriptor.Type, logger);

            return string.Join(Environment.NewLine, File.ReadAllLines(csproj.FullName).Where(line => line.Contains("<NativeAssemblyReference")));
        }
    }
}

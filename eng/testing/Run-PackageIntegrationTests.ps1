# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.

#requires -Version 7.2

[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$SelectionPath,
  [string]$WorkingDirectory = (Join-Path ([IO.Path]::GetTempPath()) "mlnet-package-tests-$([guid]::NewGuid())"),
  [string]$Filter
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repository = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$working = [IO.Path]::GetFullPath($WorkingDirectory)
if ($working.Equals($repository, [StringComparison]::OrdinalIgnoreCase) -or
    $working.StartsWith("$repository$([IO.Path]::DirectorySeparatorChar)", [StringComparison]::OrdinalIgnoreCase)) {
  throw 'Use a test directory outside the repository.'
}
if ((Test-Path -LiteralPath $working) -and @(Get-ChildItem -LiteralPath $working -Force).Count -gt 0) {
  throw 'Use an empty test directory.'
}
$diagnostics = Join-Path $working 'diagnostics'
$null = New-Item -ItemType Directory -Path $diagnostics -Force
$selectionFile = (Resolve-Path -LiteralPath $SelectionPath).Path
Copy-Item -LiteralPath $selectionFile -Destination (Join-Path $diagnostics 'selected-build.json')
$selection = Get-Content -LiteralPath $selectionFile -Raw | ConvertFrom-Json -AsHashtable
$selection.packages = [hashtable]::new($selection.packages, [StringComparer]::OrdinalIgnoreCase)
$feed = 'https://pkgs.dev.azure.com/dnceng/public/_packaging/dotnet-libraries/nuget/v3/index.json'
if ($selection.feedUrl -ne $feed) {
  throw 'Expected the public dotnet-libraries feed.'
}
[xml]$dependencies = Get-Content -LiteralPath (Join-Path $repository 'test' 'NightlyBuildDependency.props') -Raw
foreach ($reference in $dependencies.SelectNodes('/Project/ItemGroup/PackageReference')) {
  $id = $reference.GetAttribute('Include')
  if (-not $selection.packages.ContainsKey($id)) {
    throw "The selected build is missing $id."
  }
}

$versionsPath = Join-Path $working 'PublishedPackageVersions.props'
$versions = foreach ($id in $selection.packages.Keys | Sort-Object) {
  if ($id -notmatch '^[A-Za-z0-9_.-]+$' -or
      $selection.packages[$id] -notmatch '^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.+-]+)?$') {
    throw "Invalid package identity: $id."
  }
  '    <PackageVersion Remove="{0}" />' -f $id
  '    <PackageVersion Include="{0}" Version="[{1}]" />' -f $id, $selection.packages[$id]
}
"<Project>`n  <ItemGroup>`n$($versions -join "`n")`n  </ItemGroup>`n</Project>" |
  Set-Content -LiteralPath $versionsPath -Encoding utf8

# Keep the repository's build-tool feeds, but resolve selected product packages only from their publishing feed.
[xml]$nuget = Get-Content -LiteralPath (Join-Path $repository 'NuGet.config') -Raw
$nuget.SelectSingleNode('/configuration/packageSources/add[@key="dotnet-libraries"]').SetAttribute('value', $feed)
foreach ($mapping in @($nuget.SelectNodes('/configuration/packageSourceMapping/packageSource/package'))) {
  if ($selection.packages.ContainsKey($mapping.GetAttribute('pattern'))) {
    $null = $mapping.ParentNode.RemoveChild($mapping)
  }
}
$sourceMapping = $nuget.SelectSingleNode('/configuration/packageSourceMapping/packageSource[@key="dotnet-libraries"]')
foreach ($id in $selection.packages.Keys | Sort-Object) {
  $mapping = $nuget.CreateElement('package')
  $mapping.SetAttribute('pattern', $id)
  $null = $sourceMapping.AppendChild($mapping)
}
$configPath = Join-Path $working 'NuGet.config'
$nuget.Save($configPath)

$project = Join-Path $repository 'test' 'Microsoft.ML.NightlyBuild.Tests' 'Microsoft.ML.NightlyBuild.Tests.csproj'
$log = Join-Path $diagnostics 'integration-tests.log'
$properties = @(
  "-p:PublishedPackageVersions=$versionsPath"
  "-p:ArtifactsDir=$working$([IO.Path]::DirectorySeparatorChar)"
  '-p:ReferenceTypeForTestFramework=NuGet'
  '-p:UseMLCodeAnalyzer=false'
  '-p:CentralPackageTransitivePinningEnabled=false'
  '-p:TestTargetFramework=net10.0'
  '-p:Configuration=Release'
)

function Invoke-Dotnet([string[]]$Arguments) {
  & dotnet @Arguments @properties 2>&1 | Tee-Object -FilePath $log -Append | Out-Host
  if ($LASTEXITCODE -ne 0) {
    throw "dotnet $($Arguments[0]) failed with exit code $LASTEXITCODE. See $log."
  }
}

$environment = @{
  NUGET_PACKAGES = (Join-Path $working 'packages')
  ML_TEST_DATADIR = $repository
  ML_TEST_REQUIRE_NATIVE = '1'
  DOTNET_ADDITIONAL_DEPS = $null
  DOTNET_SHARED_STORE = $null
}
$original = @{}
try {
  foreach ($name in $environment.Keys) {
    $original[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    if ($null -eq $environment[$name]) {
      if (Test-Path -LiteralPath "Env:$name") {
        Remove-Item -LiteralPath "Env:$name"
      }
    }
    else {
      Set-Item -LiteralPath "Env:$name" -Value $environment[$name]
    }
  }
  "Build $($selection.buildId) ($($selection.branch)), commit $($selection.commit), results $diagnostics" |
    Tee-Object -FilePath $log | Out-Host
  Invoke-Dotnet -Arguments @('restore', $project, '--no-http-cache', '--configfile', $configPath,
    '--packages', $environment.NUGET_PACKAGES, '-p:RestoreFallbackFolders=')

  foreach ($name in @('Microsoft.ML.NightlyBuild.Tests', 'Microsoft.ML.TestFrameworkCommon')) {
    $assetsPath = Join-Path $working 'obj' $name 'project.assets.json'
    Copy-Item -LiteralPath $assetsPath -Destination (Join-Path $diagnostics "$name.assets.json")
    $assets = Get-Content -LiteralPath $assetsPath -Raw | ConvertFrom-Json -AsHashtable
    foreach ($library in $assets.libraries.Keys) {
      $id, $version = $library.Split('/', 2)
      if ($assets.libraries[$library].type -eq 'project' -and $id -ne 'Microsoft.ML.TestFrameworkCommon') {
        throw "Unexpected source project in package tests: $id."
      }
      if ($selection.packages.ContainsKey($id) -and $version -ne $selection.packages[$id]) {
        throw "Restored $library, expected $id/$($selection.packages[$id]) from build $($selection.buildId)."
      }
    }
  }

  $testArguments = @('test', $project, '--no-restore',
    '--logger', 'trx;LogFileName=package-integration.trx', '--results-directory', $diagnostics)
  if ($Filter) {
    $testArguments += @('--filter', $Filter)
  }
  Invoke-Dotnet -Arguments $testArguments
  [xml]$results = Get-Content -LiteralPath (Join-Path $diagnostics 'package-integration.trx') -Raw
  $counters = $results.SelectSingleNode("//*[local-name()='Counters']")
  if (-not $counters -or [int]$counters.GetAttribute('executed') -eq 0) {
    throw 'The package integration test run did not execute any tests.'
  }
}
finally {
  foreach ($name in $original.Keys) {
    if ($null -eq $original[$name]) {
      if (Test-Path -LiteralPath "Env:$name") {
        Remove-Item -LiteralPath "Env:$name"
      }
    }
    else {
      Set-Item -LiteralPath "Env:$name" -Value $original[$name]
    }
  }
}

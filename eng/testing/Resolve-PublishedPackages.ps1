# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.

#requires -Version 7.2

[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [ValidateRange(1, 2147483647)]
  [int]$BuildId,
  [Parameter(Mandatory = $true)]
  [string]$OutputPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$token = $env:AZURE_DEVOPS_ACCESS_TOKEN
if ([string]::IsNullOrWhiteSpace($token)) {
  $token = & az account get-access-token --resource 499b84ac-1321-427f-aa17-267ca6975798 --query accessToken --output tsv --only-show-errors
  if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($token)) {
    throw 'Sign in with Azure CLI or supply AZURE_DEVOPS_ACCESS_TOKEN with read-only official build access.'
  }
}

function Invoke-OfficialRequest([string]$Uri, [string]$OutFile) {
  $address = [uri]$Uri
  if ($address.Scheme -ne 'https' -or
      ($address.Host -ne 'dev.azure.com' -and -not $address.Host.EndsWith('.artifacts.visualstudio.com'))) {
    throw 'The official artifact service returned an unexpected download host.'
  }

  $arguments = @{
    Uri = $Uri
    Headers = @{ Authorization = "Bearer $token" }
    TimeoutSec = 60
    MaximumRetryCount = 2
    RetryIntervalSec = 2
  }
  if ($OutFile) {
    $arguments.OutFile = $OutFile
  }
  try {
    Invoke-WebRequest @arguments
  }
  catch {
    # Keep private responses and signed URLs out of the log.
    throw "Could not read build $BuildId. Check Azure DevOps access and connectivity ($($_.Exception.GetType().Name))."
  }
}

$buildApi = 'https://dev.azure.com/dnceng/internal/_apis/build'
$response = Invoke-OfficialRequest "$buildApi/builds/${BuildId}?api-version=7.1"
$build = $response.Content | ConvertFrom-Json
if ($build.id -ne $BuildId -or $build.definition.id -ne 1110 -or $build.repository.name -ne 'dotnet-machinelearning') {
  throw "Build $BuildId is not an ML.NET official build."
}
if ($build.status -ne 'completed') {
  throw "Build $BuildId has not completed."
}

$artifactResponse = Invoke-OfficialRequest "$buildApi/builds/$BuildId/artifacts?artifactName=AssetManifests&api-version=7.1"
$artifact = $artifactResponse.Content | ConvertFrom-Json
$archivePath = [IO.Path]::GetTempFileName()
try {
  $null = Invoke-OfficialRequest $artifact.resource.downloadUrl $archivePath
  $archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
  try {
    $entries = @($archive.Entries | Where-Object { $_.Name -eq 'MergedManifest.xml' })
    if ($entries.Count -ne 1) {
      throw 'Expected exactly one MergedManifest.xml in AssetManifests.'
    }
    $reader = [Xml.XmlReader]::Create($entries[0].Open(), [Xml.XmlReaderSettings]@{ CloseInput = $true })
    try {
      $manifest = [xml]::new()
      $manifest.Load($reader)
    }
    finally {
      $reader.Dispose()
    }
  }
  finally {
    $archive.Dispose()
  }
}
finally {
  Remove-Item -LiteralPath $archivePath -Force
}

$root = $manifest.DocumentElement
if ($root.LocalName -ne 'Build' -or $root.GetAttribute('AzureDevOpsBuildId') -ne [string]$BuildId -or
    $root.GetAttribute('Branch') -ne $build.sourceBranch -or
    $root.GetAttribute('Commit') -ne $build.sourceVersion) {
  throw "The package manifest does not match build $BuildId."
}
$packages = [ordered]@{}
foreach ($package in $root.SelectNodes('Package')) {
  if ($package.GetAttribute('Visibility') -ne 'External' -or $package.GetAttribute('NonShipping') -eq 'true') {
    continue
  }
  $id = $package.GetAttribute('Id')
  $version = $package.GetAttribute('Version')
  if ($id -notmatch '^[A-Za-z0-9_.-]+$' -or
      $version -notmatch '^\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.+-]+)?$' -or $packages.Contains($id)) {
    throw 'The published manifest contains invalid or duplicate package identities.'
  }
  $packages[$id] = $version
}
if (-not $packages.Contains('Microsoft.ML')) {
  throw 'The published manifest is missing Microsoft.ML.'
}

$output = [IO.Path]::GetFullPath($OutputPath)
$null = New-Item -ItemType Directory -Path (Split-Path $output -Parent) -Force
[ordered]@{
  buildId = $BuildId
  buildNumber = $build.buildNumber
  branch = $build.sourceBranch
  commit = $build.sourceVersion
  buildUrl = "https://dev.azure.com/dnceng/internal/_build/results?buildId=$BuildId"
  feedUrl = 'https://pkgs.dev.azure.com/dnceng/public/_packaging/dotnet-libraries/nuget/v3/index.json'
  packages = $packages
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $output -Encoding utf8
Write-Host "Build $BuildId ($($build.sourceBranch)): $($packages.Count) packages."

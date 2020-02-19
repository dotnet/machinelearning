// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Xml;

namespace Microsoft.ML.NugetPackageVersionUpdater
{
    class Program
    {
        private const string TempVersionsFile = "latest_versions.txt";
        private const string TargetPropsFiles = "../NightlyBuildDependency.props;../TestFrameworkDependency.props";
        private const string PackageNamespace = "Microsoft.ML";

        public static void Main(string[] args)
        {
            string projFiles = TargetPropsFiles;
            var packageVersions = GetLatestPackageVersions();
            UpdatePackageVersion(projFiles, packageVersions);
        }

        private static IDictionary<string, string> GetLatestPackageVersions()
        {
            Dictionary<string, string> packageVersions = new Dictionary<string, string>();

            using (var file = new StreamReader(TempVersionsFile))
            {
                var output = file.ReadToEnd();
                var splits = output.Split("\r\n".ToCharArray(), StringSplitOptions.RemoveEmptyEntries);
                foreach (var split in splits)
                {
                    if (split.Contains(PackageNamespace))
                    {
                        var detailSplit = split.Split(" ".ToCharArray(), StringSplitOptions.RemoveEmptyEntries);

                        //valida NuGet package version should be separate by space like below:
                        //> [PackageName]space[Requested PackageVersion]space[Resolved PackageVersion]space[Latest PackageVersion]
                        //One Example: > Microsoft.ML.LightGbm 1.4.0-preview3-28229-8 1.4.0-preview3-28229-8 1.4.0-preview3-28229-9  
                        if (detailSplit.Length == 5)
                            packageVersions.Add(detailSplit[1], detailSplit[4]);
                    }
                }
            }

            return packageVersions;
        }

        private static void UpdatePackageVersion(string projectFiles, IDictionary<string, string> latestPackageVersions)
        {
            string packageReferencePath = "/Project/ItemGroup/PackageReference";

            var projectFilePaths = projectFiles.Split(";".ToCharArray(), StringSplitOptions.RemoveEmptyEntries);

            foreach (var projectFilePath in projectFilePaths)
            {
                var csprojDoc = new XmlDocument();
                csprojDoc.Load(projectFilePath);

                var packageReferenceNodes = csprojDoc.DocumentElement.SelectNodes(packageReferencePath);

                for (int i = 0; i < packageReferenceNodes.Count; i++)
                {
                    var packageName = packageReferenceNodes.Item(i).Attributes.GetNamedItem("Include").InnerText;

                    if (latestPackageVersions.ContainsKey(packageName))
                    {
                        var latestVersion = latestPackageVersions[packageName];
                        packageReferenceNodes.Item(i).Attributes.GetNamedItem("Version").InnerText = latestVersion;
                        Console.WriteLine($"Update packege {packageName} to version {latestVersion}.");
                    }
                    else
                        Console.WriteLine($"Can't find newer version of Package {packageName} from NuGet source, don't need to update version.");
                }

                csprojDoc.Save(projectFilePath);
            }
        }
    }
}

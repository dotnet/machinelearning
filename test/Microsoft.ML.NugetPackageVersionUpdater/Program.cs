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
        //private const string getLatestVersionBatFileName = "get-latest-package-version.bat";
        private const string tempVersionsFile = "latest_versions.txt";
        private const string targetPropsFile = "..\\PackageDependency.props";
        private const string packageNamespace = "Microsoft.ML";

        public static void Main(string[] args)
        {
            string projFilePath = targetPropsFile;
            var packageVersions = GetLatestPackageVersions();
            UpdatePackageVersion(projFilePath, packageVersions);
        }

        private static IDictionary<string, string> GetLatestPackageVersions()
        {
            Dictionary<string, string> packageVersions = new Dictionary<string, string>();

            //Process p = new Process();
            //p.StartInfo.UseShellExecute = false;
            //p.StartInfo.RedirectStandardOutput = true;
            //p.StartInfo.FileName = getLatestVersionBatFileName;
            //p.Start();
            //p.WaitForExit();

            using (var file = new StreamReader(tempVersionsFile))
            {
                var output = file.ReadToEnd();
                var splits = output.Split("\r\n".ToCharArray(), StringSplitOptions.RemoveEmptyEntries);
                foreach (var split in splits)
                {
                    if (split.Contains(packageNamespace))
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

        private static void UpdatePackageVersion(string filePath, IDictionary<string, string> latestPackageVersions)
        {
            string packageReferencePath = "/Project/ItemGroup/PackageReference";

            var CsprojDoc = new XmlDocument();
            CsprojDoc.Load(filePath);

            var packageReferenceNodes = CsprojDoc.DocumentElement.SelectNodes(packageReferencePath);

            for (int i = 0; i < packageReferenceNodes.Count; i++)
            {
                var packageName = packageReferenceNodes.Item(i).Attributes.GetNamedItem("Include").InnerText;

                if (latestPackageVersions.ContainsKey(packageName))
                {
                    var latestVersion = latestPackageVersions[packageName];
                    packageReferenceNodes.Item(i).Attributes.GetNamedItem("Version").InnerText = latestVersion;
                }
                else
                    Console.WriteLine($"Can't find newer version of Package {packageName} from NuGet source, don't need to update version.");
            }

            CsprojDoc.Save(filePath);
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using System.Threading.Tasks;

namespace Microsoft.ML.Runtime.RunTests
{
    public class CopyAction
    {
        private static void CopyAll(string sourcePath, string destinationPath)
        {
            string[] directories = System.IO.Directory.GetDirectories(sourcePath, "*.*", SearchOption.AllDirectories);

            Parallel.ForEach(directories, dirPath =>
            {
                Directory.CreateDirectory(dirPath.Replace(sourcePath, destinationPath));
            });

            string[] files = System.IO.Directory.GetFiles(sourcePath, "*.*", SearchOption.AllDirectories);

            Parallel.ForEach(files, oldPath =>
            {
                var newPath = oldPath.Replace(sourcePath, destinationPath);
                FileInfo file = new FileInfo(oldPath);
                FileInfo destFile = new FileInfo(newPath);
                if (destFile.Exists)
                {
                    if (file.LastWriteTime > destFile.LastWriteTime)
                        file.CopyTo(destFile.FullName, true);
                }
                else
                    file.CopyTo(destFile.FullName, true);

            });
        }
        /// <summary>
        /// Since test folder doesn't have libraries from AutoLoad folder and Win libraries, 
        /// we copy them for each test (only newest one).
        //  This allow execute tests locally.
        /// </summary>
        public static void Execute()
        {
            var testDirectory = new FileInfo(typeof(CopyAction).Assembly.Location).Directory;
            var deploymentDirectory = testDirectory.Parent.FullName;
            if (System.Diagnostics.Process.GetCurrentProcess().ProcessName == "dotnet")
            {
                var copyfolder = "";
                if (Path.PathSeparator == '/') copyfolder = "Linux";
                else copyfolder = "Win";
                var dir = new DirectoryInfo(Path.Combine(deploymentDirectory, copyfolder));
                CopyAll(dir.FullName, testDirectory.FullName);
                CopyAll(Path.Combine(deploymentDirectory, "AutoLoad"), testDirectory.FullName);
            }
        }
    }
}

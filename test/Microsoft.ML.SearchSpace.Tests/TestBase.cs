// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using ApprovalTests;
using Xunit.Abstractions;

namespace Microsoft.ML.SearchSpace.Tests
{
    /// <summary>
    /// Test base class which provides some useful commands.
    /// </summary>
    public abstract class TestBase
    {
        public TestBase(ITestOutputHelper output)
        {
            if (Environment.GetEnvironmentVariable("HELIX_CORRELATION_ID") != null)
            {
                Approvals.UseAssemblyLocationForApprovedFiles();
            }

            Output = output;
        }

        /// <summary>
        /// Output handler.
        /// </summary>
        protected ITestOutputHelper Output { get; private set; }

        /// <summary>
        /// Get relative path of <paramref name="fileName"/> from TestData Folder.
        /// </summary>
        /// <param name="fileName">file name.</param>
        /// <returns>relative path of <paramref name="fileName"/>.</returns>
        protected string GetFileFromTestData(string fileName)
        {
            var cwd = Environment.CurrentDirectory;
            var testDataDir = new DirectoryInfo("TestData");
            return Path.Combine(testDataDir.FullName, fileName).Replace(cwd, ".");
        }

        /// <summary>
        /// Get relative path of <paramref name="folderName"/> from TestData Folder.
        /// </summary>
        /// <param name="folderName">folder name.</param>
        /// <returns>relative path.</returns>
        protected string GetFolderFromTestData(string folderName)
        {
            var cwd = Environment.CurrentDirectory;
            var testDataDir = new DirectoryInfo("TestData");
            return Path.Combine(testDataDir.FullName, folderName).Replace(cwd, ".");
        }
    }
}

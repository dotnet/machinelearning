// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.RunTests
{
    public sealed class FileSource : BaseTestClass
    {
        public FileSource(ITestOutputHelper output) : base(output)
        {
        }

        [Fact]
        public void MultiFileSourceUnitTest()
        {
            var fileSource = new MultiFileSource("adult.txt");
            Assert.True(fileSource.Count == 1);

            fileSource = new MultiFileSource("adult.tiny.with-schema.txt", "adult.tiny.with-schema.txt");
            Assert.True(fileSource.Count == 2, $"Error passing multiple paths to {nameof(MultiFileSource)}");

            //creating a directory with three files for the tests
            var dirName = Directory.CreateDirectory("MultiFileSourceUnitTest").FullName;

            var file1 = Path.Combine(dirName, "a.txt");
            var file2 = Path.Combine(dirName, "b.txt");

            File.WriteAllText(file1, "Unit Test");
            File.WriteAllText(file2, "Unit Test");

            fileSource = new MultiFileSource($"{file1}+{file2}");
            Assert.True(fileSource.Count == 2, $"Error passing concatenated paths to {nameof(MultiFileSource)}");

            fileSource = new MultiFileSource(Path.Combine(dirName, "..."));
            Assert.True(fileSource.Count == 2, $"Error passing concatenated paths to {nameof(MultiFileSource)}");

            /* Create test directories and files in the following specifications:
               /MultiFileSourceUnitTest/Data
               /MultiFileSourceUnitTest/Data/a.txt
               /MultiFileSourceUnitTest/Data/b.txt
               /MultiFileSourceUnitTest/DataFolder/
               /MultiFileSourceUnitTest/DataFolder/SubFolder1
               /MultiFileSourceUnitTest/DataFolder/SubFolder1/a.txt
               /MultiFileSourceUnitTest/DataFolder/SubFolder2
               /MultiFileSourceUnitTest/DataFolder/SubFolder2/b.txt
            */

            var dataDir = Directory.CreateDirectory("MultiFileSourceUnitTest/Data").FullName;

            var fileDataA = Path.Combine(dataDir, "a.txt");
            var fileDataB = Path.Combine(dataDir, "b.txt");

            File.WriteAllText(fileDataA, "Unit Test");
            File.WriteAllText(fileDataB, "Unit Test");

            var dataFolderDir = Directory.CreateDirectory("MultiFileSourceUnitTest/DataFolder").FullName;
            var subFolder1Dir = Directory.CreateDirectory("MultiFileSourceUnitTest/DataFolder/SubFolder1").FullName;
            var subFolder2Dir = Directory.CreateDirectory("MultiFileSourceUnitTest/DataFolder/SubFolder2").FullName;

            var fileDataSA = Path.Combine(subFolder1Dir, "a.txt");
            var fileDataSB = Path.Combine(subFolder2Dir, "b.txt");

            File.WriteAllText(fileDataSA, "Unit Test");
            File.WriteAllText(fileDataSB, "Unit Test");

            fileSource = new MultiFileSource(dataDir + "/*");
            Assert.True(fileSource.Count == 2, $"Error passing concatenated paths to {nameof(MultiFileSource)}");

            fileSource = new MultiFileSource(dataFolderDir + "/.../*");
            Assert.True(fileSource.Count == 2, $"Error passing concatenated paths to {nameof(MultiFileSource)}");

            //Delete test folder and files for test clean-up
            Directory.Delete(dirName, true);
        }
    }
}

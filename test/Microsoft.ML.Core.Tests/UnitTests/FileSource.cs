// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using System;
using System.IO;
using Xunit;

namespace Microsoft.ML.Runtime.RunTests
{
    public sealed class FileSource
    {

        [Fact]
        public void MultiFileSourceUnitTest()
        {
            var fileSource = new MultiFileSource("adult.txt");
            Assert.True(fileSource.Count == 1);

            fileSource = new MultiFileSource("adult.train", "adult.test");
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

            Assert.Throws<InvalidOperationException>(() => new MultiFileSource($"{file1}+{file2}", "adult.test"));
        }
    }
}

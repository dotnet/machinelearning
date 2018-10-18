// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.TestFramework;
using System.IO;
using Xunit;

namespace Microsoft.ML.Runtime.RunTests
{
    public class TestModelLoad
    {
        /// <summary>
        /// Tests loading a model file that was saved using an older version still loads correctly.
        /// </summary>
        [Fact]
        public void LoadOriginalBinaryLoaderModel()
        {
            using (var env = new LocalEnvironment()
                .AddStandardComponents())
            using (var modelStream = File.OpenRead(Path.Combine("TestModels", "BinaryLoader-v3.11.0.0.zip")))
            using (var rep = RepositoryReader.Open(modelStream, env))
            {
                IDataLoader result = ModelFileUtils.LoadLoader(env, rep, new MultiFileSource(null), true);

                Assert.Equal(2, result.Schema.ColumnCount);
                Assert.Equal("Image", result.Schema[0].Name);
                Assert.Equal("Class", result.Schema[1].Name);
            }
        }
    }
}
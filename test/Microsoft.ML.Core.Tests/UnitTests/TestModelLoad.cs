// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.RunTests
{
    public class TestModelLoad : BaseTestClass
    {
        public TestModelLoad(ITestOutputHelper output) : base(output)
        {
        }

        /// <summary>
        /// Tests loading a model file that was saved using version 3 (0x00010003) of BinaryLoader still loads correctly.
        /// </summary>
        [Fact]
        public void LoadBinaryLoaderModelVersion3()
        {
            var env = new MLContext(1).AddStandardComponents();
            using (var modelStream = File.OpenRead(Path.Combine("TestModels", "BinaryLoader-v3.11.0.0.zip")))
            using (var rep = RepositoryReader.Open(modelStream, env))
            {
                ILegacyDataLoader result = ModelFileUtils.LoadLoader(env, rep, new MultiFileSource(null), true);

                Assert.Equal(2, result.Schema.Count);
                Assert.Equal("Image", result.Schema[0].Name);
                Assert.Equal("Class", result.Schema[1].Name);
            }
        }

        /// <summary>
        /// Tests loading a model file containing a ConcatTransform that was saved using an older version.
        /// </summary>
        [Fact]
        public void LoadOldConcatTransformModel()
        {
            var env = new MLContext(1).AddStandardComponents();
            using (var modelStream = File.OpenRead(Path.Combine("TestModels", "ConcatTransform.zip")))
            using (var rep = RepositoryReader.Open(modelStream, env))
            {
                var result = ModelFileUtils.LoadPipeline(env, rep, new MultiFileSource(null), true);

                Assert.Equal(3, result.Schema.Count);
                Assert.Equal("Label", result.Schema[0].Name);
                Assert.Equal("Features", result.Schema[1].Name);
                Assert.Equal("Features", result.Schema[2].Name);
                Assert.Equal(9, (result.Schema[1].Type as VectorDataViewType)?.Size);
                Assert.Equal(18, (result.Schema[2].Type as VectorDataViewType)?.Size);
            }
        }
    }
}

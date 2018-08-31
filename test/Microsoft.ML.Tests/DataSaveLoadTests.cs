// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Data.IO;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.IO;
using Microsoft.ML.Runtime.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Tests
{
    public class DataSaveLoadTests : TestDataPipeBase
    {
        public DataSaveLoadTests(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void TestSaveLoadUtils()
        {
            var path = GetDataPath("iris.txt");
            using (var env = new TlcEnvironment())
            {
                var reader = new TextLoader(env, new TextLoader.Arguments(), new MultiFileSource(path));
                var data = reader.Read(new MultiFileSource(path));

                var outPath = DeleteOutputPath("saveload.txt");
                TextDataSaver.SaveData(env, data, outPath);
                IDataReader<IMultiStreamSource> newReader = new TextLoader(env, new TextLoader.Arguments(), new MultiFileSource(outPath));
                IDataView newData = newReader.Read(new MultiFileSource(outPath));
                CheckSameSchemas(data.Schema, newData.Schema);
                CheckSameValues(data, newData);

                outPath = DeleteOutputPath("saveload.idv");
                BinaryDataSaver.SaveData(env, data, outPath);
                newData = new BinaryLoader(env, new BinaryLoader.Arguments(), new MultiFileSource(outPath));
                CheckSameSchemas(data.Schema, newData.Schema);
                CheckSameValues(data, newData);
                Done();
            }
        }
    }
}

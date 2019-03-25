// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.IO;
using Microsoft.ML.Functional.Tests.Datasets;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    /// <summary>
    /// Test data input and output formats.
    /// </summary>
    public class DataIO : BaseTestClass
    {
        // Separators to test
        private readonly char[] _separators;

        public DataIO(ITestOutputHelper output) : base(output)
        {
            // SaveAsText expects a "space, tab, comma, semicolon, or bar".
            _separators = new char[] { ' ', '\t', ',', ';', '|',  };
        }

        /// <summary>
        /// Read from Enumerable: In-Memory objects can be read as enumerables into an IDatView.
        /// </summary>
        [Fact]
        public void ReadFromIEnumerable()
        {
            var mlContext = new MLContext(seed: 1);

            // Read the dataset from an enumerable.
            var data = mlContext.Data.LoadFromEnumerable(TypeTestData.GenerateDataset());

            Common.AssertTypeTestDataset(data);
        }

        /// <summary>
        /// Export to Enumerable: IDatViews can be exported as enumerables of a class.
        /// </summary>
        [Fact]
        public void ExportToIEnumerable()
        {
            var mlContext = new MLContext(seed: 1);

            // Read the dataset from an enumerable.
            var enumerableBefore = TypeTestData.GenerateDataset();
            var data = mlContext.Data.LoadFromEnumerable(enumerableBefore);

            // Export back to an enumerable.
            var enumerableAfter = mlContext.Data.CreateEnumerable<TypeTestData>(data, true);

            Common.AssertEqual(enumerableBefore, enumerableAfter);
        }

        /// <summary>
        /// Write to and read from a delimited file: Any DataKind can be written to and read from a delimited file.
        /// </summary>
        /// <remarks>
        /// Tests the roundtrip through a file using explicit schematization.
        /// </remarks>
        [Fact]
        public void WriteToAndReadFromADelimetedFile()
        {
            var mlContext = new MLContext(seed: 1);
            
            var dataBefore = mlContext.Data.LoadFromEnumerable(TypeTestData.GenerateDataset());

            foreach (var separator in _separators)
            {
                // Serialize a dataset with a known schema to a file.
                var filePath = SerializeDatasetToFile(mlContext, dataBefore, separator);
                var dataAfter = TypeTestData.GetTextLoader(mlContext, separator).Load(filePath);
                Common.AssertTestTypeDatasetsAreEqual(mlContext, dataBefore, dataAfter);
            }
        }

        /// <summary>
        /// Write to and read from a delimited file: Schematized data of any DataKind can be read from a delimited file.
        /// </summary>
        /// <remarks>
        /// Tests the roundtrip through a file using schema inference.
        /// </remarks>
        [Fact]
        public void WriteToAndReadASchemaFromADelimitedFile()
        {
            var mlContext = new MLContext(seed: 1);

            var dataBefore = mlContext.Data.LoadFromEnumerable(TypeTestData.GenerateDataset());

            foreach (var separator in _separators)
            {
                // Serialize a dataset with a known schema to a file.
                var filePath = SerializeDatasetToFile(mlContext, dataBefore, separator);
                var dataAfter = mlContext.Data.LoadFromTextFile<TypeTestData>(filePath, separatorChar: separator, hasHeader: true, allowQuoting: true);
                Common.AssertTestTypeDatasetsAreEqual(mlContext, dataBefore, dataAfter);
            }
        }

        /// <summary>
        /// Wrie to and read from a delimited file: Schematized data of any DataKind can be read from a delimited file.
        /// </summary>
        [Fact]
        public void WriteAndReadAFromABinaryFile()
        {
            var mlContext = new MLContext(seed: 1);

            var dataBefore = mlContext.Data.LoadFromEnumerable(TypeTestData.GenerateDataset());

            // Serialize a dataset with a known schema to a file.
            var filePath = SerializeDatasetToBinaryFile(mlContext, dataBefore);
            var dataAfter = mlContext.Data.LoadFromBinary(filePath);
            Common.AssertTestTypeDatasetsAreEqual(mlContext, dataBefore, dataAfter);
        }

        private string SerializeDatasetToFile(MLContext mlContext, IDataView data, char separator)
        {
            var filePath = GetOutputPath(Path.GetRandomFileName());
            using (var file = File.Create(filePath))
                mlContext.Data.SaveAsText(data, file, separatorChar: separator, headerRow: true);

            return filePath;
        }

        private string SerializeDatasetToBinaryFile(MLContext mlContext, IDataView data)
        {
            var filePath = GetOutputPath(Path.GetRandomFileName());
            using (var file = File.Create(filePath))
                mlContext.Data.SaveAsBinary(data, file);

            return filePath;
        }
    }
}

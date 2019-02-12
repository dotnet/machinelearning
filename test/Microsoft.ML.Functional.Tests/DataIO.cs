// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Functional.Tests
{
    /// <summary>
    /// Test data input and output formats
    /// </summary>
    public class DataIO : BaseTestClass
    {
        // Separators to test
        private readonly char[] _separators;

        public DataIO(ITestOutputHelper output) : base(output)
        {
            // SaveAsText expects a "space, tab, comma, semicolon, or bar"
            _separators = new char[] { ' ', '\t', ',', ';', '|',  };
        }

        /// <summary>
        /// Read from Enumerable: In-Memory objects can be read as enumerables into an IDatView.
        /// </summary>
        [Fact]
        public void ReadFromIEnumerable()
        {
            var mlContext = new MLContext(seed: 1, conc: 1);

            // Read the dataset from an enumerable
            var data = mlContext.Data.ReadFromEnumerable(GenerateToyDataset());

            ValidateToyDataset(data);
        }

        /// <summary>
        /// Export to Enumerable: IDatViews can be exported as enumerables of a class.
        /// </summary>
        [Fact]
        public void ExportToIEnumerable()
        {
            var mlContext = new MLContext(seed: 1, conc: 1);

            // Read the dataset from an enumerable
            var enumerableBefore = GenerateToyDataset();
            var data = mlContext.Data.ReadFromEnumerable(enumerableBefore);

            // Export back to an enumerable
            var enumerableAfter = mlContext.CreateEnumerable<ToyDataset>(data, true);

            AssertEqual(enumerableBefore, enumerableAfter);
        }

        /// <summary>
        /// Write to and read from a delimited file: Any DataKind can be written to and read from a delimited file.
        /// </summary>
        /// <remarks>
        /// Tests the roundtrip hrough a file using explicit schematization.
        /// </remarks>
        [Fact]
        public void WriteToAndReadFromADelimetedFile()
        {
            var mlContext = new MLContext(seed: 1, conc: 1);
            
            var dataBefore = mlContext.Data.ReadFromEnumerable(GenerateToyDataset());

            foreach (var separator in _separators)
            {
                // Serialize a dataset with a known schema to a file
                var filePath = SerializeDatasetToFile(mlContext, dataBefore, separator);
                var dataAfter = ToyDataset.GetTextLoader(mlContext, separator).Read(filePath);
                ValidateToyDataset(dataAfter);
                ToyDatasetsAreEqual(mlContext, dataBefore, dataAfter);
            }
        }

        /// <summary>
        /// Write to and read from a delimited file: Schematized data of any DataKind can be read from a delimited file.
        /// </summary>
        /// <remarks>
        /// Tests the roundtrip hrough a file using schema inference.
        /// </remarks>
        [Fact]
        public void WriteToAndReadASchemaFromADelimitedFile()
        {
            var mlContext = new MLContext(seed: 1, conc: 1);

            var dataBefore = mlContext.Data.ReadFromEnumerable(GenerateToyDataset());

            foreach (var separator in _separators)
            {
                // Serialize a dataset with a known schema to a file
                var filePath = SerializeDatasetToFile(mlContext, dataBefore, separator);
                var dataAfter = mlContext.Data.ReadFromTextFile<ToyDataset>(filePath, hasHeader: true, separatorChar: separator);
                ValidateToyDataset(dataAfter);
                ToyDatasetsAreEqual(mlContext, dataBefore, dataAfter);
            }
        }

        /// <summary>
        /// Wrie to and read from a delimited file: Schematized data of any DataKind can be read from a delimited file.
        /// </summary>
        [Fact]
        public void WriteAndReadAFromABinaryFile()
        {
            var mlContext = new MLContext(seed: 1, conc: 1);

            var dataBefore = mlContext.Data.ReadFromEnumerable(GenerateToyDataset());

            // Serialize a dataset with a known schema to a file
            var filePath = SerializeDatasetToBinaryFile(mlContext, dataBefore);
            var dataAfter = mlContext.Data.ReadFromBinary(filePath);
            ValidateToyDataset(dataAfter);
            ToyDatasetsAreEqual(mlContext, dataBefore, dataAfter);
        }

        #region FileIO
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
        #endregion

        #region ToyDataset
        private void ToyDatasetsAreEqual(MLContext mlContext, IDataView data1, IDataView data2)
        {
            // Validate that the two Schemas are the same
            Common.AssertEqual(data1.Schema, data2.Schema);

            // Define how to serialize the IDataView to objects
            var enumerable1 = mlContext.CreateEnumerable<ToyDataset>(data1, true);
            var enumerable2 = mlContext.CreateEnumerable<ToyDataset>(data2, true);

            AssertEqual(enumerable1, enumerable2);
        }

        private void AssertEqual(IEnumerable<ToyDataset> data1, IEnumerable<ToyDataset> data2)
        {
            Assert.NotNull(data1);
            Assert.NotNull(data2);
            Assert.Equal(data1.Count(), data2.Count());

            foreach (var rowPair in data1.Zip(data2, Tuple.Create))
            {
                AssertEqual(rowPair.Item1, rowPair.Item2);
            }
        }

            private void ValidateToyDataset(IDataView toyDataset)
        {
            var toyClassProperties = typeof(ToyDataset).GetProperties();

            // Check that the schema is of the right size
            Assert.Equal(17, toyDataset.Schema.Count);

            // Create a lookup table for the types and counts of all properties
            var types = new Dictionary<string, Type>();
            var counts = new Dictionary<string, int>();
            foreach (var property in toyClassProperties)
            {
                if (!property.PropertyType.IsArray)
                    types[property.Name] = property.PropertyType;
                else
                {
                    // Construct a VBuffer type for the array
                    var vBufferType = typeof(VBuffer<>);
                    Type[] typeArgs = { property.PropertyType.GetElementType() };
                    Activator.CreateInstance(property.PropertyType.GetElementType());
                    types[property.Name] = vBufferType.MakeGenericType(typeArgs);
                }
                    
                counts[property.Name] = 0;
            }

            foreach (var column in toyDataset.Schema)
            {
                Assert.True(types.ContainsKey(column.Name));
                Assert.Equal(1, ++counts[column.Name]);
                Assert.Equal(types[column.Name], column.Type.RawType);
            }

            // Make sure we didn't miss any columns
            foreach (var value in counts.Values)
                Assert.Equal(1, value);
        }

        private IEnumerable<ToyDataset> GenerateToyDataset(int numExamples = 5, int seed = 1)
        {
            var rng = new Random(seed);
            for (int i = 0; i < numExamples; i++)
            {
                yield return new ToyDataset
                {
                    Label = rng.NextDouble() > 0.5,
                    Features = new float[] {
                        (float)rng.NextDouble(),
                        (float)rng.NextDouble(),
                        (float)rng.NextDouble(),
                        (float)rng.NextDouble(),
                        (float)rng.NextDouble()
                    },
                    I1 = (sbyte)rng.Next(),
                    U1 = (byte)rng.Next(),
                    I2 = (short)rng.Next(),
                    U2 = (ushort)rng.Next(),
                    I4 = rng.Next(),
                    U4 = (uint)rng.Next(),
                    I8 = (long)rng.Next(),
                    U8 = (ulong)rng.Next(),
                    R4 = (float)rng.NextDouble(),
                    R8 = (double)rng.NextDouble(),
                    Tx = GetRandomRomChar(rng),
                    Ts = TimeSpan.FromSeconds(rng.NextDouble() * (1+rng.Next())),
                    Dt = DateTime.FromOADate(rng.Next(657435, 2958465)),
                    Dz = DateTimeOffset.FromUnixTimeSeconds((long)(rng.NextDouble() * (1 + rng.Next()))),
                    Ug = new RowId((ulong)rng.Next(), (ulong)rng.Next())
                };
            }
        }

        private ReadOnlyMemory<char> GetRandomRomChar(Random rng, int length = 10)
        {
            var chars = new char[length];
            for (int i = 0; i < length; i++)
                chars[i] = (char)(32 + rng.Next(0, 94)); // From space to ~
            return new ReadOnlyMemory<char>(chars);
        }

        private sealed class ToyDataset
        {
            [LoadColumn(0)]
            public bool Label { get; set; }

            [LoadColumn(1, 5), VectorType(5)]
            public float[] Features { get; set; }

            [LoadColumn(6)]
            public sbyte I1 { get; set; }

            [LoadColumn(7)]
            public byte U1 { get; set; }

            [LoadColumn(8)]
            public short I2 { get; set; }

            [LoadColumn(9)]
            public ushort U2 { get; set; }

            [LoadColumn(10)]
            public int I4 { get; set; }

            [LoadColumn(11)]
            public uint U4 { get; set; }

            [LoadColumn(12)]
            public long I8 { get; set; }

            [LoadColumn(13)]
            public ulong U8 { get; set; }

            [LoadColumn(14)]
            public float R4 { get; set; }

            [LoadColumn(15)]
            public double R8 { get; set; }

            [LoadColumn(16)]
            public ReadOnlyMemory<char> Tx { get; set; }

            [LoadColumn(17)]
            public TimeSpan Ts { get; set; }

            [LoadColumn(18)]
            public DateTime Dt { get; set; }

            [LoadColumn(19)]
            public DateTimeOffset Dz { get; set; }

            [LoadColumn(20)]
            public RowId Ug { get; set; }

            public static TextLoader GetTextLoader(MLContext mlContext, char separator)
            {
                return mlContext.Data.CreateTextLoader(
                        new[] {
                            new TextLoader.Column("Label", DataKind.Bool, 0),
                            new TextLoader.Column("Features", DataKind.R4, 1, 5),
                            new TextLoader.Column("I1", DataKind.I1, 6),
                            new TextLoader.Column("U1", DataKind.U1, 7),
                            new TextLoader.Column("I2", DataKind.I2, 8),
                            new TextLoader.Column("U2", DataKind.U2, 9),
                            new TextLoader.Column("I4", DataKind.I4, 10),
                            new TextLoader.Column("U4", DataKind.U4, 11),
                            new TextLoader.Column("I8", DataKind.I8, 12),
                            new TextLoader.Column("U8", DataKind.U8, 13),
                            new TextLoader.Column("R4", DataKind.R4, 14),
                            new TextLoader.Column("R8", DataKind.R8, 15),
                            new TextLoader.Column("Tx", DataKind.TX, 16),
                            new TextLoader.Column("Ts", DataKind.TS, 17),
                            new TextLoader.Column("Dt", DataKind.DT, 18),
                            new TextLoader.Column("Dz", DataKind.DZ, 19),
                            new TextLoader.Column("Ug", DataKind.UG, 20),
                        },
                        hasHeader: true,
                        separatorChar: separator);
            }
        }

        private static void AssertEqual(ToyDataset toyDataset1, ToyDataset toyDataset2)
        {
            Assert.Equal(toyDataset1.Label, toyDataset2.Label);
            Common.AssertEqual(toyDataset1.Features, toyDataset2.Features);
            Assert.Equal(toyDataset1.I1, toyDataset2.I1);
            Assert.Equal(toyDataset1.U1, toyDataset2.U1);
            Assert.Equal(toyDataset1.I2, toyDataset2.I2);
            Assert.Equal(toyDataset1.U2, toyDataset2.U2);
            Assert.Equal(toyDataset1.I4, toyDataset2.I4);
            Assert.Equal(toyDataset1.U4, toyDataset2.U4);
            Assert.Equal(toyDataset1.I8, toyDataset2.I8);
            Assert.Equal(toyDataset1.U8, toyDataset2.U8);
            Assert.Equal(toyDataset1.R4, toyDataset2.R4);
            Assert.Equal(toyDataset1.R8, toyDataset2.R8);
            Assert.Equal(toyDataset1.Tx.ToString(), toyDataset2.Tx.ToString());
            Assert.True(toyDataset1.Ts.Equals(toyDataset2.Ts));
            Assert.True(toyDataset1.Dt.Equals(toyDataset2.Dt));
            Assert.True(toyDataset1.Dz.Equals(toyDataset2.Dz));
            Assert.True(toyDataset1.Ug.Equals(toyDataset2.Ug));
        }
        #endregion
    }
}

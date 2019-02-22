// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;

namespace Microsoft.ML.Functional.Tests.Datasets
{
    /// <summary>
    /// A class containing one property per <see cref="DataKind"/>.
    /// </summary>
    /// <remarks>
    /// This class has annotations for automatic deserialization from a file, and contains helper methods
    /// for reading from a file and for generating a random dataset as an IEnumerable.
    /// </remarks>
    internal sealed class TypeTestData
    {
        private const int _numFeatures = 10;

        [LoadColumn(0)]
        public bool Label { get; set; }

        [LoadColumn(1)]
        public sbyte I1 { get; set; }

        [LoadColumn(2)]
        public byte U1 { get; set; }

        [LoadColumn(3)]
        public short I2 { get; set; }

        [LoadColumn(4)]
        public ushort U2 { get; set; }

        [LoadColumn(5)]
        public int I4 { get; set; }

        [LoadColumn(6)]
        public uint U4 { get; set; }

        [LoadColumn(7)]
        public long I8 { get; set; }

        [LoadColumn(8)]
        public ulong U8 { get; set; }

        [LoadColumn(9)]
        public float R4 { get; set; }

        [LoadColumn(10)]
        public double R8 { get; set; }

        [LoadColumn(11)]
        public ReadOnlyMemory<char> Tx { get; set; }

        [LoadColumn(12)]
        public TimeSpan Ts { get; set; }

        [LoadColumn(13)]
        public DateTime Dt { get; set; }

        [LoadColumn(14)]
        public DateTimeOffset Dz { get; set; }

        [LoadColumn(15)]
        public DataViewRowId Ug { get; set; }

        [LoadColumn(16, 16 + _numFeatures - 1), VectorType(_numFeatures)]
        public float[] Features { get; set; }


        /// <summary>
        /// Get the text loader for the <see cref="TypeTestData"/> dataset.
        /// </summary>
        /// <param name="mlContext">The ML Context.</param>
        /// <param name="separator">The Separator to read with.</param>
        /// <returns></returns>
        public static TextLoader GetTextLoader(MLContext mlContext, char separator)
        {
            return mlContext.Data.CreateTextLoader(
                    new[] {
                        new TextLoader.Column("Label", DataKind.Bool, 0),
                        new TextLoader.Column("I1", DataKind.I1, 1),
                        new TextLoader.Column("U1", DataKind.U1, 2),
                        new TextLoader.Column("I2", DataKind.I2, 3),
                        new TextLoader.Column("U2", DataKind.U2, 4),
                        new TextLoader.Column("I4", DataKind.I4, 5),
                        new TextLoader.Column("U4", DataKind.U4, 6),
                        new TextLoader.Column("I8", DataKind.I8, 7),
                        new TextLoader.Column("U8", DataKind.U8, 8),
                        new TextLoader.Column("R4", DataKind.R4, 9),
                        new TextLoader.Column("R8", DataKind.R8, 10),
                        new TextLoader.Column("Tx", DataKind.TX, 11),
                        new TextLoader.Column("Ts", DataKind.TS, 12),
                        new TextLoader.Column("Dt", DataKind.DT, 13),
                        new TextLoader.Column("Dz", DataKind.DZ, 14),
                        new TextLoader.Column("Ug", DataKind.UG, 15),
                        new TextLoader.Column("Features", DataKind.R4, 16, 16 + _numFeatures-1),
                    },
                    separatorChar: separator,
                    hasHeader: true,
                    allowQuoting: true);
        }

        /// <summary>
        /// Generate an IEnumerable of <see cref="TypeTestData"/>.
        /// </summary>
        /// <param name="numExamples">The number of <see cref="TypeTestData"/> objects to make.</param>
        /// <param name="seed">The random seed.</param>
        /// <returns>An IEnumerable of <see cref="TypeTestData"/>.</returns>
        public static IEnumerable<TypeTestData> GenerateDataset(int numExamples = 5, int seed = 1)
        {
            var rng = new Random(seed);
            for (int i = 0; i < numExamples; i++)
            {
                yield return GetRandomInstance(rng);
            }
        }

        /// <summary>
        /// Get a random instance of <see cref="TypeTestData"/>.
        /// </summary>
        /// <param name="rng">A <see cref="Random"/> object.</param>
        /// <returns></returns>
        public static TypeTestData GetRandomInstance(Random rng)
        {
            if (rng == null)
                throw new ArgumentNullException("rng");

            return new TypeTestData
            {
                Label = rng.NextDouble() > 0.5,
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
                Tx = GetRandomCharSpan(rng),
                Ts = TimeSpan.FromSeconds(rng.NextDouble() * (1 + rng.Next())),
                Dt = DateTime.FromOADate(rng.Next(657435, 2958465)),
                Dz = DateTimeOffset.FromUnixTimeSeconds((long)(rng.NextDouble() * (1 + rng.Next()))),
                Ug = new DataViewRowId((ulong)rng.Next(), (ulong)rng.Next()),
                Features = GetRandomFloatArray(rng, _numFeatures),
            };
        }

        private static ReadOnlyMemory<char> GetRandomCharSpan(Random rng, int length = 10)
        {
            var chars = new char[length];
            for (int i = 0; i < length; i++)
                chars[i] = (char)(32 + rng.Next(0, 94)); // From space to ~.
            return new ReadOnlyMemory<char>(chars);
        }

        private static float[] GetRandomFloatArray(Random rng, int length)
        {
            var floatArray = new float[length];
            for (int i = 0; i < length; i++)
                floatArray[i] = (float)rng.NextDouble();
            return floatArray;
        }
    }
}

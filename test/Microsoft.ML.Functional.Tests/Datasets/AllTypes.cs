// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Xunit;

namespace Microsoft.ML.Functional.Tests.Datasets
{
    internal sealed class AllTypes
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

        /// <summary>
        /// Get the text loader for the AllTypes dataset.
        /// </summary>
        /// <param name="mlContext">The ML Context.</param>
        /// <param name="separator">The Separator to read with.</param>
        /// <returns></returns>
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

        /// <summary>
        /// Generate an IEnumerable of AllTypes.
        /// </summary>
        /// <param name="numExamples">The number of AllTypesDataset objects to make.</param>
        /// <param name="seed">The random seed.</param>
        /// <returns>An IEnumerable of AllTypes.</returns>
        public static IEnumerable<AllTypes> GenerateDataset(int numExamples = 5, int seed = 1)
        {
            var rng = new Random(seed);
            for (int i = 0; i < numExamples; i++)
            {
                yield return new AllTypes
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
                    Tx = GetRandomCharSpan(rng),
                    Ts = TimeSpan.FromSeconds(rng.NextDouble() * (1 + rng.Next())),
                    Dt = DateTime.FromOADate(rng.Next(657435, 2958465)),
                    Dz = DateTimeOffset.FromUnixTimeSeconds((long)(rng.NextDouble() * (1 + rng.Next()))),
                    Ug = new RowId((ulong)rng.Next(), (ulong)rng.Next())
                };
            }
        }

        private static ReadOnlyMemory<char> GetRandomCharSpan(Random rng, int length = 10)
        {
            var chars = new char[length];
            for (int i = 0; i < length; i++)
                chars[i] = (char)(32 + rng.Next(0, 94)); // From space to ~.
            return new ReadOnlyMemory<char>(chars);
        }
    }
}

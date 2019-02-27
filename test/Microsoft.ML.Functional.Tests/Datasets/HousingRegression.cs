// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Functional.Tests.Datasets
{
    /// <summary>
    /// A schematized class for loading the HousingRegression dataset.
    /// </summary>
    internal sealed class HousingRegression
    {
        [LoadColumn(0), ColumnName("Label")]
        public float MedianHomeValue { get; set; }

        [LoadColumn(1)]
        public float CrimesPerCapita { get; set; }

        [LoadColumn(2)]
        public float PercentResidental { get; set; }

        [LoadColumn(3)]
        public float PercentNonRetail { get; set; }

        [LoadColumn(4)]
        public float CharlesRiver { get; set; }

        [LoadColumn(5)]
        public float NitricOxides { get; set; }

        [LoadColumn(6)]
        public float RoomsPerDwelling { get; set; }

        [LoadColumn(7)]
        public float PercentPre40s { get; set; }

        [LoadColumn(8)]
        public float EmploymentDistance { get; set; }

        [LoadColumn(9)]
        public float HighwayDistance { get; set; }

        [LoadColumn(10)]
        public float TaxRate { get; set; }

        [LoadColumn(11)]
        public float TeacherRatio { get; set; }

        /// <summary>
        /// The list of columns commonly used as features
        /// </summary>
        public static readonly string[] Features = new string[] {"CrimesPerCapita", "PercentResidental", "PercentNonRetail", "CharlesRiver", "NitricOxides",
                "RoomsPerDwelling", "PercentPre40s", "EmploymentDistance", "HighwayDistance", "TaxRate", "TeacherRatio"};
    }
}

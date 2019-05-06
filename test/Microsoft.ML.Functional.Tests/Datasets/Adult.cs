// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML.Functional.Tests.Datasets
{
    /// <summary>
    /// A class for the Adult test dataset.
    /// </summary>
    internal sealed class Adult
    {
        [LoadColumn(0)]
        public bool Label { get; set; }

        [LoadColumn(1)]
        public string WorkClass { get; set; }

        [LoadColumn(2)]
        public string Education { get; set; }

        [LoadColumn(3)]
        public string MaritalStatus { get; set; }

        [LoadColumn(4)]
        public string Occupation { get; set; }

        [LoadColumn(5)]
        public string Relationship { get; set; }

        [LoadColumn(6)]
        public string Ethnicity { get; set; }

        [LoadColumn(7)]
        public string Sex { get; set; }

        [LoadColumn(8)]
        public string NativeCountryRegion { get; set; }

        [LoadColumn(9)]
        public float Age { get; set; }

        [LoadColumn(10)]
        public float FinalWeight { get; set; }

        [LoadColumn(11)]
        public float EducationNum { get; set; }

        [LoadColumn(12)]
        public float CapitalGain { get; set; }

        [LoadColumn(13)]
        public float CapitalLoss { get; set; }

        [LoadColumn(14)]
        public float HoursPerWeek { get; set; }

        /// <summary>
        /// The list of columns commonly used as categorical features.
        /// </summary>
        public static readonly string[] CategoricalFeatures = new string[] { "WorkClass", "Education", "MaritalStatus", "Occupation", "Relationship", "Ethnicity", "Sex", "NativeCountryRegion" };

        /// <summary>
        /// The list of columns commonly used as numerical features.
        /// </summary>
        public static readonly string[] NumericalFeatures = new string[] { "Age", "FinalWeight", "EducationNum", "CapitalGain", "CapitalLoss", "HoursPerWeek" };
    }
}

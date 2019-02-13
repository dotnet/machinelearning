// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Data.DataView;
using Microsoft.ML.Data;
using Microsoft.ML.Functional.Tests.Datasets;
using Xunit;

namespace Microsoft.ML.Functional.Tests
{
    internal static class Common
    {
        /// <summary>
        /// Asssert that an <see cref="IDataView"/> rows are of <see cref="AllTypes"/>.
        /// </summary>
        /// <param name="allTypesDataset">An <see cref="IDataView"/>.</param>
        public static void AssertAllTypesDataset(IDataView allTypesDataset)
        {
            var toyClassProperties = typeof(AllTypes).GetProperties();

            // Check that the schema is of the right size.
            Assert.Equal(toyClassProperties.Length, allTypesDataset.Schema.Count);

            // Create a lookup table for the types and counts of all properties.
            var types = new Dictionary<string, Type>();
            var counts = new Dictionary<string, int>();
            foreach (var property in toyClassProperties)
            {
                if (!property.PropertyType.IsArray)
                    types[property.Name] = property.PropertyType;
                else
                {
                    // Construct a VBuffer type for the array.
                    var vBufferType = typeof(VBuffer<>);
                    Type[] typeArgs = { property.PropertyType.GetElementType() };
                    Activator.CreateInstance(property.PropertyType.GetElementType());
                    types[property.Name] = vBufferType.MakeGenericType(typeArgs);
                }

                counts[property.Name] = 0;
            }

            foreach (var column in allTypesDataset.Schema)
            {
                Assert.True(types.ContainsKey(column.Name));
                Assert.Equal(1, ++counts[column.Name]);
                Assert.Equal(types[column.Name], column.Type.RawType);
            }

            // Make sure we didn't miss any columns.
            foreach (var value in counts.Values)
                Assert.Equal(1, value);
        }

        /// <summary>
        /// Assert than two <see cref="AllTypes"/> datasets are equal.
        /// </summary>
        /// <param name="mlContext">The ML Context.</param>
        /// <param name="data1">A <see cref="IDataView"/> of <see cref="AllTypes"/></param>
        /// <param name="data2">A <see cref="IDataView"/> of <see cref="AllTypes"/></param>
        public static void AssertAllTypesDatasetsAreEqual(MLContext mlContext, IDataView data1, IDataView data2)
        {
            // Confirm that they are both of the propery row type.
            AssertAllTypesDataset(data1);
            AssertAllTypesDataset(data2);

            // Validate that the two Schemas are the same.
            Common.AssertEqual(data1.Schema, data2.Schema);

            // Define how to serialize the IDataView to objects.
            var enumerable1 = mlContext.CreateEnumerable<AllTypes>(data1, true);
            var enumerable2 = mlContext.CreateEnumerable<AllTypes>(data2, true);

            AssertEqual(enumerable1, enumerable2);
        }

        /// <summary>
        /// Assert that two float arrays are equal.
        /// </summary>
        /// <param name="array1">An array of floats.</param>
        /// <param name="array2">An array of floats.</param>
        public static void AssertEqual(float[] array1, float[] array2)
        {
            Assert.NotNull(array1);
            Assert.NotNull(array2);
            Assert.Equal(array1.Length, array2.Length);

            for (int i = 0; i < array1.Length; i++)
                Assert.Equal(array1[i], array2[i]);
        }

        /// <summary>
        ///  Assert that two <see cref="Schema"/> objects are equal.
        /// </summary>
        /// <param name="schema1">A <see cref="Schema"/> object.</param>
        /// <param name="schema2">A <see cref="Schema"/> object.</param>
        public static void AssertEqual(Schema schema1, Schema schema2)
        {
            Assert.NotNull(schema1);
            Assert.NotNull(schema2);

            Assert.Equal(schema1.Count(), schema2.Count());

            foreach (var schemaPair in schema1.Zip(schema2, Tuple.Create))
            {
                Assert.Equal(schemaPair.Item1.Name, schemaPair.Item2.Name);
                Assert.Equal(schemaPair.Item1.Index, schemaPair.Item2.Index);
                Assert.Equal(schemaPair.Item1.IsHidden, schemaPair.Item2.IsHidden);
                // Can probably do a better comparison of Metadata.
                AssertEqual(schemaPair.Item1.Metadata.Schema, schemaPair.Item1.Metadata.Schema);
                Assert.True((schemaPair.Item1.Type == schemaPair.Item2.Type) ||
                    (schemaPair.Item1.Type.RawType == schemaPair.Item2.Type.RawType));
            }
        }

        /// <summary>
        /// Assert than two <see cref="AllTypes"/> enumerables are equal.
        /// </summary>
        /// <param name="data1">An enumerable of <see cref="AllTypes"/></param>
        /// <param name="data2">An enumerable of <see cref="AllTypes"/></param>
        public static void AssertEqual(IEnumerable<AllTypes> data1, IEnumerable<AllTypes> data2)
        {
            Assert.NotNull(data1);
            Assert.NotNull(data2);
            Assert.Equal(data1.Count(), data2.Count());

            foreach (var rowPair in data1.Zip(data2, Tuple.Create))
            {
                AssertEqual(rowPair.Item1, rowPair.Item2);
            }
        }

        /// <summary>
        /// Assert that two AllTypes datasets are equal.
        /// </summary>
        /// <param name="allTypes1">An <see cref="AllTypes"/>.</param>
        /// <param name="allTypes2">An <see cref="AllTypes"/>.</param>
        public static void AssertEqual(AllTypes allTypes1, AllTypes allTypes2)
        {
            Assert.Equal(allTypes1.Label, allTypes2.Label);
            Common.AssertEqual(allTypes1.Features, allTypes2.Features);
            Assert.Equal(allTypes1.I1, allTypes2.I1);
            Assert.Equal(allTypes1.U1, allTypes2.U1);
            Assert.Equal(allTypes1.I2, allTypes2.I2);
            Assert.Equal(allTypes1.U2, allTypes2.U2);
            Assert.Equal(allTypes1.I4, allTypes2.I4);
            Assert.Equal(allTypes1.U4, allTypes2.U4);
            Assert.Equal(allTypes1.I8, allTypes2.I8);
            Assert.Equal(allTypes1.U8, allTypes2.U8);
            Assert.Equal(allTypes1.R4, allTypes2.R4);
            Assert.Equal(allTypes1.R8, allTypes2.R8);
            Assert.Equal(allTypes1.Tx.ToString(), allTypes2.Tx.ToString());
            Assert.True(allTypes1.Ts.Equals(allTypes2.Ts));
            Assert.True(allTypes1.Dt.Equals(allTypes2.Dt));
            Assert.True(allTypes1.Dz.Equals(allTypes2.Dz));
            Assert.True(allTypes1.Ug.Equals(allTypes2.Ug));
        }

        /// <summary>
        /// Check that a <see cref="RegressionMetrics"/> object is valid.
        /// </summary>
        /// <param name="metrics">The metrics object.</param>
        public static void CheckMetrics(RegressionMetrics metrics)
        {
            // Perform sanity checks on the metrics.
            Assert.True(metrics.Rms >= 0);
            Assert.True(metrics.L1 >= 0);
            Assert.True(metrics.L2 >= 0);
            Assert.True(metrics.RSquared <= 1);
        }
    }
}

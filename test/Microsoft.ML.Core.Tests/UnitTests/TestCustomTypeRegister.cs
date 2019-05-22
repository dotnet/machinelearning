// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.RunTests
{
    public class TestCustomTypeRegister : TestDataViewBase
    {
        public TestCustomTypeRegister(ITestOutputHelper helper)
           : base(helper)
        {
        }

        /// <summary>
        /// A custom type which ML.NET doesn't know yet. Its value will be loaded as a DataView column in this test.
        /// </summary>
        private class AlienBody
        {
            public int Age { get; set; }
            public float Height { get; set; }
            public float Weight { get; set; }
            public int HandCount { get; set; }

            public AlienBody(int age, float height, float weight, int handCount)
            {
                Age = age;
                Height = height;
                Weight = weight;
                HandCount = handCount;
            }
        }

        [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
        private sealed class AlienTypeAttributeAttribute : Attribute
        {
            public int Id { get; }

            /// <summary>
            /// Create an image type with known height and width.
            /// </summary>
            public AlienTypeAttributeAttribute(int id)
            {
                Id = id;
            }
        }

        /// <summary>
        /// A custom class with a type which ML.NET doesn't know yet. Its value will be loaded as a DataView row in this test.
        /// It will be the input of <see cref="AlienLambda.MergeBody(AlienHero, SuperAlienHero)"/>.
        ///
        /// <see cref="One"/> and <see cref="Two"/> would be mapped to different types inside ML.NET type system because they
        /// have different <see cref="AlienTypeAttributeAttribute"/>s.
        /// </summary>
        private class AlienHero
        {
            public string Name { get; set; }

            [AlienTypeAttribute(100)]
            public AlienBody One { get; set; }

            [AlienTypeAttribute(200)]
            public AlienBody Two { get; set; }

            public AlienHero()
            {
                Name = "Unknown";
                One = new AlienBody(0, 0, 0, 0);
                Two = new AlienBody(0, 0, 0, 0);
            }

            public AlienHero(string name,
                int age, float height, float weight, int handCount,
                int anotherAge, float anotherHeight, float anotherWeight, int anotherHandCount)
            {
                Name = "Unknown";
                One = new AlienBody(age, height, weight, handCount);
                Two = new AlienBody(anotherAge, anotherHeight, anotherWeight, anotherHandCount);
            }
        }

        /// <summary>
        /// Type of <see cref="AlienBody"/> in ML.NET's type system.
        /// </summary>
        private class DataViewAlienBodyType : StructuredDataViewType
        {
            public int Id { get; }

            public DataViewAlienBodyType(int id) : base(typeof(AlienBody))
            {
                Id = id;
            }

            public override bool Equals(DataViewType other)
            {
                if (other is DataViewAlienBodyType)
                    return ((DataViewAlienBodyType)other).Id == Id;
                else
                    return false;
            }

            public override int GetHashCode()
            {
                return Id.GetHashCode();
            }
        }

        /// <summary>
        /// The output type of processing <see cref="AlienHero"/> using <see cref="AlienLambda.MergeBody(AlienHero, SuperAlienHero)"/>.
        /// </summary>
        private class SuperAlienHero
        {
            public string Name { get; set; }

            [AlienTypeAttribute(007)]
            public AlienBody Merged { get; set; }

            public SuperAlienHero()
            {
                Name = "Unknown";
                Merged = new AlienBody(0, 0, 0, 0);
            }
        }

        /// <summary>
        /// A mapping from <see cref="AlienHero"/> to <see cref="SuperAlienHero"/>. It is used to create a
        /// <see cref="CustomMappingEstimator{TSrc, TDst}"/> in <see cref="RegisterTypeWithAttribute()"/>.
        /// </summary>
        [CustomMappingFactoryAttribute("LambdaAlienHero")]
        private class AlienLambda : CustomMappingFactory<AlienHero, SuperAlienHero>
        {
            public static void MergeBody(AlienHero input, SuperAlienHero output)
            {
                output.Name = "Super " + input.Name;
                output.Merged.Age = input.One.Age + input.Two.Age;
                output.Merged.Height = input.One.Height + input.Two.Height;
                output.Merged.Weight = input.One.Weight + input.Two.Weight;
                output.Merged.HandCount = input.One.HandCount + input.Two.HandCount;
            }

            public override Action<AlienHero, SuperAlienHero> GetMapping()
            {
                return MergeBody;
            }
        }

        [Fact]
        public void RegisterTypeWithAttribute()
        {
            var tribe = new List<AlienHero>() { new AlienHero("ML.NET", 2, 1000, 2000, 3000, 4000, 5000, 6000, 7000) };

            // Type manager doesn't know any of those custom types, so all calls to it should return false.
            Assert.False(DataViewTypeManager.Knows(new DataViewAlienBodyType(100)));
            Assert.False(DataViewTypeManager.Knows(new DataViewAlienBodyType(200)));
            Assert.False(DataViewTypeManager.Knows(new DataViewAlienBodyType(007)));
            Assert.False(DataViewTypeManager.Knows(typeof(AlienBody), new[] { new AlienTypeAttributeAttribute(100) }));
            Assert.False(DataViewTypeManager.Knows(typeof(AlienBody), new[] { new AlienTypeAttributeAttribute(200) }));
            Assert.False(DataViewTypeManager.Knows(typeof(AlienBody), new[] { new AlienTypeAttributeAttribute(007) }));

            // Register those custom types.
            DataViewTypeManager.Register(new DataViewAlienBodyType(100), typeof(AlienBody), new[] { new AlienTypeAttributeAttribute(100) });
            DataViewTypeManager.Register(new DataViewAlienBodyType(200), typeof(AlienBody), new[] { new AlienTypeAttributeAttribute(200) });
            DataViewTypeManager.Register(new DataViewAlienBodyType(007), typeof(AlienBody), new[] { new AlienTypeAttributeAttribute(007) });

            // Type manager now knows those custom types, so all calls to it should return true.
            Assert.True(DataViewTypeManager.Knows(new DataViewAlienBodyType(100)));
            Assert.True(DataViewTypeManager.Knows(new DataViewAlienBodyType(200)));
            Assert.True(DataViewTypeManager.Knows(new DataViewAlienBodyType(007)));
            Assert.True(DataViewTypeManager.Knows(typeof(AlienBody), new[] { new AlienTypeAttributeAttribute(100) }));
            Assert.True(DataViewTypeManager.Knows(typeof(AlienBody), new[] { new AlienTypeAttributeAttribute(200) }));
            Assert.True(DataViewTypeManager.Knows(typeof(AlienBody), new[] { new AlienTypeAttributeAttribute(007) }));

            // Check if the custom type (AlienBody with its attributes) is registered correctly with a DataView type (DataViewAlienBodyType).
            Assert.Equal(new DataViewAlienBodyType(100),
                DataViewTypeManager.GetDataViewType(typeof(AlienBody), new[] { new AlienTypeAttributeAttribute(100) }));
            Assert.Equal(new DataViewAlienBodyType(200),
                DataViewTypeManager.GetDataViewType(typeof(AlienBody), new[] { new AlienTypeAttributeAttribute(200) }));
            Assert.Equal(new DataViewAlienBodyType(007),
                DataViewTypeManager.GetDataViewType(typeof(AlienBody), new[] { new AlienTypeAttributeAttribute(007) }));

            // Build a ML.NET pipeline and make prediction.
            var tribeDataView = ML.Data.LoadFromEnumerable(tribe);

            var heroEstimator = new CustomMappingEstimator<AlienHero, SuperAlienHero>(ML, AlienLambda.MergeBody, "LambdaAlienHero");

            var model = heroEstimator.Fit(tribeDataView);

            var tribeTransformed = model.Transform(tribeDataView);

            var tribeEnumerable = ML.Data.CreateEnumerable<SuperAlienHero>(tribeTransformed, false).ToList();

            // Make sure the pipeline output is correct.
            Assert.Equal(tribeEnumerable[0].Name, "Super " + tribe[0].Name);
            Assert.Equal(tribeEnumerable[0].Merged.Age, tribe[0].One.Age + tribe[0].Two.Age);
            Assert.Equal(tribeEnumerable[0].Merged.Height, tribe[0].One.Height + tribe[0].Two.Height);
            Assert.Equal(tribeEnumerable[0].Merged.Weight, tribe[0].One.Weight + tribe[0].Two.Weight);
            Assert.Equal(tribeEnumerable[0].Merged.HandCount, tribe[0].One.HandCount + tribe[0].Two.HandCount);

            // Build prediction engine from the trained pipeline.
            var engine = ML.Model.CreatePredictionEngine<AlienHero, SuperAlienHero>(model);
            var alien = new AlienHero("TEN.LM", 1, 2, 3, 4, 5, 6, 7, 8);
            var superAlien = engine.Predict(alien);

            // Make sure the prediction engine produces expected result.
            Assert.Equal(superAlien.Name, "Super " + alien.Name);
            Assert.Equal(superAlien.Merged.Age, alien.One.Age + alien.Two.Age);
            Assert.Equal(superAlien.Merged.Height, alien.One.Height + alien.Two.Height);
            Assert.Equal(superAlien.Merged.Weight, alien.One.Weight + alien.Two.Weight);
            Assert.Equal(superAlien.Merged.HandCount, alien.One.HandCount + alien.Two.HandCount);
        }

        [Fact]
        public void TestTypeManager()
        {
            // Semantically identical DataViewTypes should produce the same hash code.
            var a = new DataViewAlienBodyType(9527);
            var aCode = a.GetHashCode();
            var b = new DataViewAlienBodyType(9527);
            var bCode = b.GetHashCode();

            Assert.Equal(aCode, bCode);

            // Semantically identical attributes should produce the same hash code.
            var c = new AlienTypeAttributeAttribute(1228);
            var cCode = c.GetHashCode();
            var d = new AlienTypeAttributeAttribute(1228);
            var dCode = d.GetHashCode();

            Assert.Equal(cCode, dCode);

            // Check registering the same type pair is OK.
            DataViewTypeManager.Register(a, typeof(AlienBody));
            DataViewTypeManager.Register(a, typeof(AlienBody));

            // Make sure registering the same type twice throws.
            bool isWrong = false;
            try
            {
                // "a" has been registered with AlienBody without any attribute, so the user can't
                // register "a" again with AlienBody plus the attribute "c."
                DataViewTypeManager.Register(a, typeof(AlienBody), new[] { c });
            }
            catch
            {
                isWrong = true;
            }
            Assert.True(isWrong);

            // Make sure registering the same type twice throws.
            bool isWrongAgain = false;
            try
            {
                // AlienBody has been registered with "a," so user can't register it with
                // "new DataViewAlienBodyType(5566)" again.
                DataViewTypeManager.Register(new DataViewAlienBodyType(5566), typeof(AlienBody));
            }
            catch
            {
                isWrongAgain = true;
            }
            Assert.True(isWrongAgain);
        }
    }
}

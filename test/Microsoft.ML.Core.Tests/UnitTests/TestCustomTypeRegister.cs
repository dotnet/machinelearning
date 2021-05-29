// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
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

        /// <summary>
        /// <see cref="DataViewTypeAttribute"/> applied to class <see cref="AlienBody"/> members.
        /// </summary>
        private sealed class AlienTypeAttributeAttribute : DataViewTypeAttribute
        {
            public int RaceId { get; }

            /// <summary>
            /// Create an <see cref="DataViewTypeAttribute"/> from <paramref name="raceId"/> to a <see cref="AlienBody"/>.
            /// </summary>
            public AlienTypeAttributeAttribute(int raceId)
            {
                RaceId = raceId;
            }

            /// <summary>
            /// A function implicitly invoked by ML.NET when processing a custom type. It binds a DataViewType to a custom type plus its attributes.
            /// </summary>
            public override void Register()
            {
                DataViewTypeManager.Register(new DataViewAlienBodyType(RaceId), typeof(AlienBody), this);
            }

            public override bool Equals(DataViewTypeAttribute other)
            {
                if (other is AlienTypeAttributeAttribute alienTypeAttributeAttribute)
                    return RaceId == alienTypeAttributeAttribute.RaceId;
                return false;
            }

            public override int GetHashCode() => RaceId.GetHashCode();
        }

        /// <summary>
        /// A custom class with a type which ML.NET doesn't know yet. Its value will be loaded as a DataView row in this test.
        /// It will be the input of <see cref="AlienFusionProcess.MergeBody(AlienHero, SuperAlienHero)"/>.
        ///
        /// <see cref="One"/> and <see cref="Two"/> would be mapped to different types inside ML.NET type system because they
        /// have different <see cref="AlienTypeAttributeAttribute"/>s. For example, the column type of <see cref="One"/> would
        /// be <see cref="DataViewAlienBodyType"/>.
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
        /// It usually shows up as <see cref="DataViewSchema.Column.Type"/> among <see cref="IDataView.Schema"/>.
        /// </summary>
        private class DataViewAlienBodyType : StructuredDataViewType
        {
            public int RaceId { get; }

            public DataViewAlienBodyType(int id) : base(typeof(AlienBody))
            {
                RaceId = id;
            }

            public override bool Equals(DataViewType other)
            {
                if (other is DataViewAlienBodyType otherAlien)
                    return otherAlien.RaceId == RaceId;
                return false;
            }

            public override int GetHashCode()
            {
                return RaceId.GetHashCode();
            }
        }

        /// <summary>
        /// The output type of processing <see cref="AlienHero"/> using <see cref="AlienFusionProcess.MergeBody(AlienHero, SuperAlienHero)"/>.
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
        /// <see cref="CustomMappingEstimator{TSrc, TDst}"/> in <see cref="RegisterTypeWithAttribute(bool)"/>.
        /// </summary>
        [CustomMappingFactoryAttribute("LambdaAlienHero")]
        private class AlienFusionProcess : CustomMappingFactory<AlienHero, SuperAlienHero>
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

        [Theory]
        [InlineData(true)]
        [InlineData(false)]
        public void RegisterTypeWithAttribute(bool saveModel)
        {
            // Build in-memory data.
            var tribe = new List<AlienHero>() { new AlienHero("ML.NET", 2, 1000, 2000, 3000, 4000, 5000, 6000, 7000) };

            // Build a ML.NET pipeline and make prediction.
            var tribeDataView = ML.Data.LoadFromEnumerable(tribe);
            var heroEstimator = new CustomMappingEstimator<AlienHero, SuperAlienHero>(ML, AlienFusionProcess.MergeBody, "LambdaAlienHero");
            var model = heroEstimator.Fit(tribeDataView);
            var tribeTransformed = model.Transform(tribeDataView);
            var tribeEnumerable = ML.Data.CreateEnumerable<SuperAlienHero>(tribeTransformed, false).ToList();

            ITransformer modelForPrediction = model;
            if (saveModel)
            {
                ML.Model.Save(model, tribeDataView.Schema, "customTransform.zip");
                modelForPrediction = ML.Model.Load("customTransform.zip", out var tribeDataViewSchema);
            }

            // Make sure the pipeline output is correct.
            Assert.Equal(tribeEnumerable[0].Name, "Super " + tribe[0].Name);
            Assert.Equal(tribeEnumerable[0].Merged.Age, tribe[0].One.Age + tribe[0].Two.Age);
            Assert.Equal(tribeEnumerable[0].Merged.Height, tribe[0].One.Height + tribe[0].Two.Height);
            Assert.Equal(tribeEnumerable[0].Merged.Weight, tribe[0].One.Weight + tribe[0].Two.Weight);
            Assert.Equal(tribeEnumerable[0].Merged.HandCount, tribe[0].One.HandCount + tribe[0].Two.HandCount);

            // Build prediction engine from the trained pipeline.
            var engine = ML.Model.CreatePredictionEngine<AlienHero, SuperAlienHero>(modelForPrediction);
            var alien = new AlienHero("TEN.LM", 1, 2, 3, 4, 5, 6, 7, 8);
            var superAlien = engine.Predict(alien);

            // Make sure the prediction engine produces expected result.
            Assert.Equal(superAlien.Name, "Super " + alien.Name);
            Assert.Equal(superAlien.Merged.Age, alien.One.Age + alien.Two.Age);
            Assert.Equal(superAlien.Merged.Height, alien.One.Height + alien.Two.Height);
            Assert.Equal(superAlien.Merged.Weight, alien.One.Weight + alien.Two.Weight);
            Assert.Equal(superAlien.Merged.HandCount, alien.One.HandCount + alien.Two.HandCount);

            Done();
        }

        [Fact]
        void TestCustomTransformBackcompat()
        {
            // With older versions, it is necessary to register the assembly
            ML.ComponentCatalog.RegisterAssembly(typeof(AlienFusionProcess).Assembly);

            var modelPath = Path.Combine(DataDir, "backcompat", "customTransform.zip");
            var trainedModel = ML.Model.Load(modelPath, out var dataViewSchema);

            var engine = ML.Model.CreatePredictionEngine<AlienHero, SuperAlienHero>(trainedModel);
            var alien = new AlienHero("TEN.LM", 1, 2, 3, 4, 5, 6, 7, 8);
            var superAlien = engine.Predict(alien);

            // Make sure the prediction engine produces expected result.
            Assert.Equal(superAlien.Name, "Super " + alien.Name);
            Assert.Equal(superAlien.Merged.Age, alien.One.Age + alien.Two.Age);
            Assert.Equal(superAlien.Merged.Height, alien.One.Height + alien.Two.Height);
            Assert.Equal(superAlien.Merged.Weight, alien.One.Weight + alien.Two.Weight);
            Assert.Equal(superAlien.Merged.HandCount, alien.One.HandCount + alien.Two.HandCount);

            Done();
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
            // Note that "a" and "b" should be identical.
            DataViewTypeManager.Register(a, typeof(AlienBody));
            DataViewTypeManager.Register(a, typeof(AlienBody));
            DataViewTypeManager.Register(b, typeof(AlienBody));
            DataViewTypeManager.Register(b, typeof(AlienBody));

            // Check if register of (a, typeof(AlienBody)) successes.
            Assert.True(DataViewTypeManager.Knows(a));
            Assert.True(DataViewTypeManager.Knows(b));
            Assert.True(DataViewTypeManager.Knows(typeof(AlienBody)));
            Assert.Equal(a, DataViewTypeManager.GetDataViewType(typeof(AlienBody)));
            Assert.Equal(b, DataViewTypeManager.GetDataViewType(typeof(AlienBody)));

            // Make sure registering the same type twice throws.
            bool isWrong = false;
            try
            {
                // "a" has been registered with AlienBody without any attribute, so the user can't
                // register "a" again with AlienBody plus the attribute "c."
                DataViewTypeManager.Register(a, typeof(AlienBody), c);
            }
            catch
            {
                isWrong = true;
            }
            Assert.True(isWrong);

            // Make sure registering the same type twice throws.
            isWrong = false;
            try
            {
                // AlienBody has been registered with "a," so user can't register it with
                // "new DataViewAlienBodyType(5566)" again.
                DataViewTypeManager.Register(new DataViewAlienBodyType(5566), typeof(AlienBody));
            }
            catch
            {
                isWrong = true;
            }
            Assert.True(isWrong);

            // Register a type with attribute.
            var e = new DataViewAlienBodyType(7788);
            var f = new AlienTypeAttributeAttribute(8877);
            DataViewTypeManager.Register(e, typeof(AlienBody), f);
            Assert.True(DataViewTypeManager.Knows(e));
            Assert.True(DataViewTypeManager.Knows(typeof(AlienBody), new[] { f }));
            // "e" is associated with typeof(AlienBody) with "f," so the call below should return true.
            Assert.Equal(e, DataViewTypeManager.GetDataViewType(typeof(AlienBody), new[] { f }));
            // "a" is associated with typeof(AlienBody) without any attribute, so the call below should return false.
            Assert.NotEqual(a, DataViewTypeManager.GetDataViewType(typeof(AlienBody), new[] { f }));
        }

        [Fact]
        public void GetTypeWithAdditionalDataViewTypeAttributes()
        {
            var a = new DataViewAlienBodyType(7788);
            var b = new AlienTypeAttributeAttribute(8877);
            var c = new ColumnNameAttribute("foo");
            var d = new AlienTypeAttributeAttribute(8876);


            DataViewTypeManager.Register(a, typeof(AlienBody), b);
            Assert.True(DataViewTypeManager.Knows(a));
            Assert.True(DataViewTypeManager.Knows(typeof(AlienBody), new Attribute[] { b, c }));
            // "a" is associated with typeof(AlienBody) with "b," so the call below should return true.
            Assert.Equal(a, DataViewTypeManager.GetDataViewType(typeof(AlienBody), new Attribute[] { b, c }));
            Assert.Throws<ArgumentOutOfRangeException>(() => DataViewTypeManager.Knows(typeof(AlienBody), new Attribute[] { b, d }));
        }
    }
}

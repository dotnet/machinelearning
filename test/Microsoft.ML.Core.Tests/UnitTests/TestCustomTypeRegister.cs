// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
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
        private class Body
        {
            public int Age { get; set; }
            public float Height { get; set; }
            public float Weight { get; set; }

            /// <summary>
            /// Type register should happen before the creation of the first <see cref="Body"/>. Otherwise, ML.NET might not recognize
            /// that <see cref="Body"/> is typed to <see cref="DataViewBodyType"/> in ML.NET's internal type system.
            /// </summary>
            static Body()
            {
                DataViewTypeManager.Register(typeof(Body), DataViewBodyType.Instance);
            }

            public Body()
            {
                Age = 0;
                Height = 0;
                Weight = 0;
            }

            public Body(int age, float height, float weight)
            {
                Age = age;
                Height = height;
                Weight = weight;
            }
        }

        /// <summary>
        /// A custom class with a type which ML.NET doesn't know yet. Its value will be loaded as a DataView row in this test.
        /// </summary>
        private class Hero
        {
            public string Name { get; set; }
            public Body One { get; set; }

            public Hero()
            {
                Name = "Earth";
                One = new Body(10000000, 500000, 800000);
            }

            public Hero(string name, int age, float height, float weight)
            {
                Name = name;
                One = new Body(age, height, weight);
            }
        }

        /// <summary>
        /// Type of <see cref="Body"/> in ML.NET.
        /// </summary>
        private class DataViewBodyType : DataViewType
        {
            private static volatile DataViewBodyType _instance;

            /// <summary>
            /// The singleton instance of this type.
            /// </summary>
            public static DataViewBodyType Instance
            {
                get
                {
                    return _instance ??
                        Interlocked.CompareExchange(ref _instance, new DataViewBodyType(), null) ??
                        _instance;
                }
            }

            private DataViewBodyType() : base(typeof(Body))
            {
            }

            public override bool Equals(DataViewType other)
            {
                if (other == this)
                    return true;
                return false;
            }
        }

        /// <summary>
        /// Pass in <see cref="Body"/> as a column in <see cref="IDataView"/> and load <see cref="Body"/> back.
        /// </summary>
        [Fact]
        public void RegisterCustomType()
        {
            var tribe = new List<Hero>(){ new Hero("Earth", 10, 5.8f, 100.0f), new Hero("Mars", 20, 6.8f, 120.8f) };

            var tribeDataView = ML.Data.LoadFromEnumerable(tribe);
            var tribeEnumerable = ML.Data.CreateEnumerable<Hero>(tribeDataView, false).ToList();

            for (int i = 0; i < tribe.Count; ++i)
            {
                Assert.Equal(tribe[i].Name, tribeEnumerable[i].Name);
                Assert.Equal(tribe[i].One.Age, tribeEnumerable[i].One.Age);
                Assert.Equal(tribe[i].One.Height, tribeEnumerable[i].One.Height);
                Assert.Equal(tribe[i].One.Weight, tribeEnumerable[i].One.Weight);
            }
        }

        private class SuperHero
        {
            public string SuperName { get; set; }
            public Body SuperOne { get; set; }

            public SuperHero()
            {
                SuperName = "IronMan";
                SuperOne = new Body();
            }
        }

        [CustomMappingFactoryAttribute("LambdaHero")]
        private class MyLambda : CustomMappingFactory<Hero, SuperHero>
        {
            public static void Grow(Hero input, SuperHero output)
            {
                output.SuperName = "Sr. " + input.Name;
                output.SuperOne.Age = input.One.Age + 9999;
                output.SuperOne.Height = input.One.Height * 10;
                output.SuperOne.Weight = input.One.Weight * 10;
            }

            public override Action<Hero, SuperHero> GetMapping()
            {
                return Grow;
            }
        }

        [Fact]
        public void ModifyCustomType()
        {
            var tribe = new List<Hero>(){ new Hero("Earth", 10, 5.8f, 100.0f) };

            var tribeDataView = ML.Data.LoadFromEnumerable(tribe);

            var heroEstimator = new CustomMappingEstimator<Hero, SuperHero>(ML, MyLambda.Grow, "LambdaHero");

            var tribeTransformed = heroEstimator.Fit(tribeDataView).Transform(tribeDataView);

            var tribeEnumerable = ML.Data.CreateEnumerable<SuperHero>(tribeTransformed, false).ToList();

            for (int i = 0; i < tribe.Count; ++i)
            {
                Assert.Equal("Sr. " + tribe[i].Name, tribeEnumerable[i].SuperName);
                Assert.Equal(tribe[i].One.Age + 9999, tribeEnumerable[i].SuperOne.Age);
                Assert.Equal(tribe[i].One.Height * 10, tribeEnumerable[i].SuperOne.Height);
                Assert.Equal(tribe[i].One.Weight * 10, tribeEnumerable[i].SuperOne.Weight);
            }
        }
    }
}

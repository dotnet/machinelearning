using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    class CustomMappingWithInMemoryCustomType
    {
        // This example shows how custom mapping actions can be performed on custom data
        // types that ML.NET doesn't know yet. The example tells a story of how two alien
        // bodies are merged to form a super alien with a single body.
        //
        // Here, the type AlienHero represents a single alien entity with a member "Name"
        // of type string and members "One" and "Two" of type AlienBody. It defines a custom
        // mapping action AlienFusionProcess that takes an AlienHero and "fuses" its two
        // AlienBody members to produce a SuperAlienHero entity with a "Name" member of type
        // string and a single "Merged" member of type AlienBody, where the merger is just
        // the addition of the various members of AlienBody.
        public static void Example()
        {
            var mlContext = new MLContext();
            // Build in-memory data.
            var tribe = new List<AlienHero>() { new AlienHero("ML.NET", 2, 1000,
                2000, 3000, 4000, 5000, 6000, 7000) };

            // Build a ML.NET pipeline and make prediction.
            var tribeDataView = mlContext.Data.LoadFromEnumerable(tribe);
            var pipeline = mlContext.Transforms.CustomMapping(AlienFusionProcess
                .GetMapping(), contractName: null);

            var model = pipeline.Fit(tribeDataView);
            var tribeTransformed = model.Transform(tribeDataView);

            // Print out prediction produced by the model.
            var firstAlien = mlContext.Data.CreateEnumerable<SuperAlienHero>(
                tribeTransformed, false).First();

            Console.WriteLine("We got a super alien with name " + firstAlien.Name +
                ", age " + firstAlien.Merged.Age + ", " + "height " + firstAlien
                .Merged.Height + ", weight  " + firstAlien.Merged.Weight + ", and "
                + firstAlien.Merged.HandCount + " hands.");

            // Expected output:
            //   We got a super alien with name Super ML.NET, age 4002, height 6000, weight 8000, and 10000 hands.

            // Create a prediction engine and print out its prediction.
            var engine = mlContext.Model.CreatePredictionEngine<AlienHero,
                SuperAlienHero>(model);

            var alien = new AlienHero("TEN.LM", 1, 2, 3, 4, 5, 6, 7, 8);
            var superAlien = engine.Predict(alien);
            Console.Write("We got a super alien with name " + superAlien.Name +
                ", age " + superAlien.Merged.Age + ", height " +
                superAlien.Merged.Height + ", weight " + superAlien.Merged.Weight +
                ", and " + superAlien.Merged.HandCount + " hands.");

            // Expected output:
            //   We got a super alien with name Super TEN.LM, age 6, height 8, weight 10, and 12 hands.
        }

        // A custom type which ML.NET doesn't know yet. Its value will be loaded as
        // a DataView column in this example.
        //
        // The type members represent the characteristics of an alien body that will
        // be merged in the AlienFusionProcess.
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

        // DataViewTypeAttribute applied to class AlienBody members. This attribute
        // defines how class AlienBody is registered in ML.NET's type system. In this
        // case, AlienBody is registered as DataViewAlienBodyType in ML.NET. The RaceId
        // property allows different members of type AlienBody to be registered with
        // different types in ML.NEt (see usage in class AlienHero).
        private sealed class AlienTypeAttributeAttribute : DataViewTypeAttribute
        {
            public int RaceId { get; }

            // Create an DataViewTypeAttribute> from raceId to a AlienBody.
            public AlienTypeAttributeAttribute(int raceId)
            {
                RaceId = raceId;
            }

            // A function implicitly invoked by ML.NET when processing a custom
            // type. It binds a DataViewType to a custom type plus its attributes.
            public override void Register()
            {
                DataViewTypeManager.Register(new DataViewAlienBodyType(RaceId),
                    typeof(AlienBody), this);
            }

            public override bool Equals(DataViewTypeAttribute other)
            {
                if (other is AlienTypeAttributeAttribute alienTypeAttributeAttribute)
                    return RaceId == alienTypeAttributeAttribute.RaceId;
                return false;
            }

            public override int GetHashCode() => RaceId.GetHashCode();
        }

        // A custom class with a type which ML.NET doesn't know yet. Its value will
        // be loaded as a DataView row in this example. It will be the input of
        // AlienFusionProcess.MergeBody(AlienHero, SuperAlienHero).
        //
        // The members One and Two would be mapped to different types inside
        // ML.NET type system because they have different 
        // AlienTypeAttributeAttribute's. For example, the column type of One would
        // be DataViewAlienBodyType with RaceId=100.
        //
        // This type represents a "Hero" Alien that is a single entity with two bodies.
        // The "Hero" undergoes a fusion process defined in AlienFusionProcess to
        // become a SuperAlienHero with a single body that is a merger of the two
        // bodies.
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
                int anotherAge, float anotherHeight, float anotherWeight, int
                    anotherHandCount)
            {
                Name = name;
                One = new AlienBody(age, height, weight, handCount);
                Two = new AlienBody(anotherAge, anotherHeight, anotherWeight,
                    anotherHandCount);
            }
        }

        // Type of AlienBody in ML.NET's type system. This is the data view type that
        // will represent AlienBody in ML.NET's type system when it is registered as
        // such in AlienTypeAttributeAttribute.
        // It usually shows up as DataViewSchema.Column.Type among IDataView.Schema.
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

        // The output type of processing AlienHero using AlienFusionProcess
        // .MergeBody(AlienHero, SuperAlienHero).
        // This is a "fused" alien whose body is a merger of the two bodies
        // of AlienHero.
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

        // The implementation of custom mapping is MergeBody. It accepts AlienHero
        // and produces SuperAlienHero.
        private class AlienFusionProcess
        {
            public static void MergeBody(AlienHero input, SuperAlienHero output)
            {
                output.Name = "Super " + input.Name;
                output.Merged.Age = input.One.Age + input.Two.Age;
                output.Merged.Height = input.One.Height + input.Two.Height;
                output.Merged.Weight = input.One.Weight + input.Two.Weight;
                output.Merged.HandCount = input.One.HandCount + input.Two.HandCount;
            }

            public static Action<AlienHero, SuperAlienHero> GetMapping()
            {
                return MergeBody;
            }
        }
    }
}

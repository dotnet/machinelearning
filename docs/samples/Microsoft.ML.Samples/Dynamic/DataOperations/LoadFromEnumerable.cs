using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Samples.Dynamic
{
    public static class LoadFromEnumerable
    {
        // Creating IDataView from IEnumerable, and setting the size of the vector
        // at runtime. When the data model is defined through types, setting the
        // size of the vector is done through the VectorType annotation. When the
        // size of the data is not known at compile time, the Schema can be directly
        // modified at runtime and the size of the vector set there. This is
        // important, because most of the ML.NET trainers require the Features
        // vector to be of known size. 
        public static void Example()
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness.
            var mlContext = new MLContext();

            // Get a small dataset as an IEnumerable.
            IEnumerable<DataPointVector> enumerableKnownSize = new DataPointVector[]
            {
               new DataPointVector{ Features = new float[]{ 1.2f, 3.4f, 4.5f, 3.2f,
                   7,5f } },

               new DataPointVector{ Features = new float[]{ 4.2f, 3.4f, 14.65f,
                   3.2f, 3,5f } },

               new DataPointVector{ Features = new float[]{ 1.6f, 3.5f, 4.5f, 6.2f,
                   3,5f } },

            };

            // Load dataset into an IDataView. 
            IDataView data = mlContext.Data.LoadFromEnumerable(enumerableKnownSize);
            var featureColumn = data.Schema["Features"].Type as VectorDataViewType;
            // Inspecting the schema
            Console.WriteLine($"Is the size of the Features column known: " +
                $"{featureColumn.IsKnownSize}.\nSize: {featureColumn.Size}");

            // Preview
            //
            // Is the size of the Features column known? True.
            // Size: 5.

            // If the size of the vector is unknown at compile time, it can be set 
            // at runtime.
            IEnumerable<DataPoint> enumerableUnknownSize = new DataPoint[]
            {
               new DataPoint{ Features = new float[]{ 1.2f, 3.4f, 4.5f } },
               new DataPoint{ Features = new float[]{ 4.2f, 3.4f, 1.6f } },
               new DataPoint{ Features = new float[]{ 1.6f, 3.5f, 4.5f } },
            };

            // The feature dimension (typically this will be the Count of the array 
            // of the features vector known at runtime).
            int featureDimension = 3;
            var definedSchema = SchemaDefinition.Create(typeof(DataPoint));
            featureColumn = definedSchema["Features"]
                .ColumnType as VectorDataViewType;

            Console.WriteLine($"Is the size of the Features column known: " +
                $"{featureColumn.IsKnownSize}.\nSize: {featureColumn.Size}");

            // Preview
            //
            // Is the size of the Features column known? False.
            // Size: 0.

            // Set the column type to be a known-size vector.
            var vectorItemType = ((VectorDataViewType)definedSchema[0].ColumnType)
                .ItemType;
            definedSchema[0].ColumnType = new VectorDataViewType(vectorItemType,
                featureDimension);

            // Read the data into an IDataView with the modified schema supplied in
            IDataView data2 = mlContext.Data
                .LoadFromEnumerable(enumerableUnknownSize, definedSchema);

            featureColumn = data2.Schema["Features"].Type as VectorDataViewType;
            // Inspecting the schema
            Console.WriteLine($"Is the size of the Features column known: " +
                $"{featureColumn.IsKnownSize}.\nSize: {featureColumn.Size}");

            // Preview
            //
            // Is the size of the Features column known? True. 
            // Size: 3.
        }
    }

    public class DataPoint
    {
        public float[] Features { get; set; }
    }

    public class DataPointVector
    {
        [VectorType(5)]
        public float[] Features { get; set; }
    }
}

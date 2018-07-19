// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.TestFramework;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.EntryPoints.Tests
{
    public class CollectionDataSourceTests : BaseTestClass
    {
        public CollectionDataSourceTests(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        public void CheckConstructor()
        {
            Assert.NotNull(CollectionDataSource.Create(new List<Input>() { new Input { Number1 = 1, String1 = "1" } }));
            Assert.NotNull(CollectionDataSource.Create(new Input[1] { new Input { Number1 = 1, String1 = "1" } }));
            Assert.NotNull(CollectionDataSource.Create(new Input[1] { new Input { Number1 = 1, String1 = "1" } }.AsEnumerable()));

            bool thrown = false;
            try
            {
                CollectionDataSource.Create(new List<Input>());
            }
            catch
            {
                thrown = true;
            }
            Assert.True(thrown);

            thrown = false;
            try
            {
                CollectionDataSource.Create(new Input[0]);
            }
            catch
            {
                thrown = true;
            }
            Assert.True(thrown);
        }

        [Fact]
        public void CanSuccessfullyApplyATransform()
        {
            var collection = CollectionDataSource.Create(new List<Input>() { new Input { Number1 = 1, String1 = "1" } });
            using (var environment = new TlcEnvironment())
            {
                Experiment experiment = environment.CreateExperiment();
                ILearningPipelineDataStep output = (ILearningPipelineDataStep)collection.ApplyStep(null, experiment);

                Assert.NotNull(output.Data);
                Assert.NotNull(output.Data.VarName);
                Assert.Null(output.Model);
            }
        }

        [Fact]
        public void CanSuccessfullyEnumerated()
        {
            var collection = CollectionDataSource.Create(new List<Input>() {
                new Input { Number1 = 1, String1 = "1" },
                new Input { Number1 = 2, String1 = "2" },
                new Input { Number1 = 3, String1 = "3" }
            });

            using (var environment = new TlcEnvironment())
            {
                Experiment experiment = environment.CreateExperiment();
                ILearningPipelineDataStep output = collection.ApplyStep(null, experiment) as ILearningPipelineDataStep;

                experiment.Compile();
                collection.SetInput(environment, experiment);
                experiment.Run();

                IDataView data = experiment.GetOutput(output.Data);
                Assert.NotNull(data);

                using (var cursor = data.GetRowCursor((a => true)))
                {
                    var IDGetter = cursor.GetGetter<float>(0);
                    var TextGetter = cursor.GetGetter<DvText>(1);

                    Assert.True(cursor.MoveNext());

                    float ID = 0;
                    IDGetter(ref ID);
                    Assert.Equal(1, ID);

                    DvText Text = new DvText();
                    TextGetter(ref Text);
                    Assert.Equal("1", Text.ToString());

                    Assert.True(cursor.MoveNext());

                    ID = 0;
                    IDGetter(ref ID);
                    Assert.Equal(2, ID);

                    Text = new DvText();
                    TextGetter(ref Text);
                    Assert.Equal("2", Text.ToString());

                    Assert.True(cursor.MoveNext());

                    ID = 0;
                    IDGetter(ref ID);
                    Assert.Equal(3, ID);

                    Text = new DvText();
                    TextGetter(ref Text);
                    Assert.Equal("3", Text.ToString());

                    Assert.False(cursor.MoveNext());
                }
            }
        }

        [Fact]
        public void CanTrain()
        {
            var pipeline = new LearningPipeline();
            var data = new List<IrisData>() {
                new IrisData { SepalLength = 1f, SepalWidth = 1f, PetalLength=0.3f, PetalWidth=5.1f, Label=1},
                new IrisData { SepalLength = 1f, SepalWidth = 1f, PetalLength=0.3f, PetalWidth=5.1f, Label=1},
                new IrisData { SepalLength = 1.2f, SepalWidth = 0.5f, PetalLength=0.3f, PetalWidth=5.1f, Label=0}
            };
            var collection = CollectionDataSource.Create(data);

            pipeline.Add(collection);
            pipeline.Add(new ColumnConcatenator(outputColumn: "Features",
                "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            PredictionModel<IrisData, IrisPrediction> model = pipeline.Train<IrisData, IrisPrediction>();

            IrisPrediction prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });

            pipeline = new LearningPipeline();
            collection = CollectionDataSource.Create(data.AsEnumerable());
            pipeline.Add(collection);
            pipeline.Add(new ColumnConcatenator(outputColumn: "Features",
                "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            model = pipeline.Train<IrisData, IrisPrediction>();

            prediction = model.Predict(new IrisData()
            {
                SepalLength = 3.3f,
                SepalWidth = 1.6f,
                PetalLength = 0.2f,
                PetalWidth = 5.1f,
            });

        }

        public class Input
        {
            [Column("0")]
            public float Number1;

            [Column("1")]
            public string String1;
        }

        public class IrisData
        {
            [Column("0")]
            public float Label;

            [Column("1")]
            public float SepalLength;

            [Column("2")]
            public float SepalWidth;

            [Column("3")]
            public float PetalLength;

            [Column("4")]
            public float PetalWidth;
        }

        public class IrisPrediction
        {
            [ColumnName("Score")]
            public float[] PredictedLabels;
        }

        public class ConversionSimpleClass
        {
            public int fInt;
            public uint fuInt;
            public short fShort;
            public ushort fuShort;
            public sbyte fsByte;
            public byte fByte;
            public long fLong;
            public ulong fuLong;
            public float fFloat;
            public double fDouble;
            public bool fBool;
            public string fString;
        }

        public class ConversionNullalbeClass
        {
            public int? fInt;
            public uint? fuInt;
            public short? fShort;
            public ushort? fuShort;
            public sbyte? fsByte;
            public byte? fByte;
            public long? fLong;
            public ulong? fuLong;
            public float? fFloat;
            public double? fDouble;
            public bool? fBool;
            public string fString;
        }

        public bool CompareObjectValues(object x, object y, Type type)
        {
            //handle string conversion.
            //by default behaviour for DvText is to be empty string, while for string is null.
            //so if we do roundtrip string-> DvText -> string all null string become empty strings.
            //therefore replace all null values to empty string if field is string.
            if (type == typeof(string) && x == null)
                x = "";
            if (type == typeof(string) && y == null)
                y = "";
            if (x == null && y == null)
                return true;
            if (x == null && y != null)
                return false;
            return (x.Equals(y));
        }

        public bool CompareThrougReflection<T>(T x, T y)
        {
            foreach (var field in typeof(T).GetFields())
            {
                var xvalue = field.GetValue(x);
                var yvalue = field.GetValue(y);
                if (field.FieldType.IsArray)
                {
                    if (!CompareArrayValues(xvalue as Array, yvalue as Array))
                        return false;
                }
                else
                {
                    if (!CompareObjectValues(xvalue, yvalue, field.FieldType))
                        return false;
                }

            }
            return true;
        }

        public bool CompareArrayValues(Array x, Array y)
        {
            if (x == null && y == null) return true;
            if ((x == null && y != null) || (y == null && x != null))
                return false;
            if (x.Length != y.Length)
                return false;
            for (int i = 0; i < x.Length; i++)
                if (!CompareObjectValues(x.GetValue(i), y.GetValue(i), x.GetType().GetElementType()))
                    return false;
            return true;
        }

        public class ClassWithConstField
        {
            public const string ConstString = "N";
            public string fString;
            public const int ConstInt = 100;
            public int fInt;
        }

        [Fact]
        public void BackAndForthConversionWithBasicTypes()
        {
            var data = new List<ConversionSimpleClass>()
            {
                new ConversionSimpleClass(){ fInt=int.MaxValue-1, fuInt=uint.MaxValue-1, fBool=true, fsByte=sbyte.MaxValue-1, fByte = byte.MaxValue-1,
                    fDouble =double.MaxValue-1, fFloat=float.MaxValue-1, fLong=long.MaxValue-1, fuLong = ulong.MaxValue-1,
                    fShort =short.MaxValue-1, fuShort = ushort.MaxValue-1, fString=null},
                new ConversionSimpleClass(){ fInt=int.MaxValue, fuInt=uint.MaxValue, fBool=true, fsByte=sbyte.MaxValue, fByte = byte.MaxValue,
                    fDouble =double.MaxValue, fFloat=float.MaxValue, fLong=long.MaxValue, fuLong = ulong.MaxValue,
                   fShort =short.MaxValue, fuShort = ushort.MaxValue, fString="ooh"},
                new ConversionSimpleClass(){ fInt=int.MinValue+1, fuInt=uint.MinValue+1, fBool=true, fsByte=sbyte.MinValue+1, fByte = byte.MinValue+1,
                    fDouble =double.MinValue+1, fFloat=float.MinValue+1, fLong=long.MinValue+1, fuLong = ulong.MinValue+1,
                    fShort =short.MinValue+1, fuShort = ushort.MinValue+1, fString=""},
                new ConversionSimpleClass(){},
            };

            var dataNullable = new List<ConversionNullalbeClass>()
            {
                new ConversionNullalbeClass(){ fInt=int.MaxValue-1, fuInt=uint.MaxValue-1, fBool=true, fsByte=sbyte.MaxValue-1, fByte = byte.MaxValue-1,
                    fDouble =double.MaxValue-1, fFloat=float.MaxValue-1, fLong=long.MaxValue-1, fuLong = ulong.MaxValue-1,
                    fShort =short.MaxValue-1, fuShort = ushort.MaxValue-1, fString="ha"},
                new ConversionNullalbeClass(){ fInt=int.MaxValue, fuInt=uint.MaxValue, fBool=true, fsByte=sbyte.MaxValue, fByte = byte.MaxValue,
                    fDouble =double.MaxValue, fFloat=float.MaxValue, fLong=long.MaxValue, fuLong = ulong.MaxValue,
                    fShort =short.MaxValue, fuShort = ushort.MaxValue, fString="ooh"},
                 new ConversionNullalbeClass(){ fInt=int.MinValue+1, fuInt=uint.MinValue+1, fBool=true, fsByte=sbyte.MinValue+1, fByte = byte.MinValue+1,
                    fDouble =double.MinValue+1, fFloat=float.MinValue+1, fLong=long.MinValue+1, fuLong = ulong.MinValue+1,
                    fShort =short.MinValue+1, fuShort = ushort.MinValue+1, fString=""},
                new ConversionNullalbeClass()
            };

            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var enumeratorSimple = dataView.AsEnumerable<ConversionSimpleClass>(env, false).GetEnumerator();
                var originalEnumerator = data.GetEnumerator();
                while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
                {
                    Assert.True(CompareThrougReflection(enumeratorSimple.Current, originalEnumerator.Current));
                }
                Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());

                dataView = ComponentCreation.CreateDataView(env, dataNullable);
                var enumeratorNullable = dataView.AsEnumerable<ConversionNullalbeClass>(env, false).GetEnumerator();
                var originalNullableEnumerator = dataNullable.GetEnumerator();
                while (enumeratorNullable.MoveNext() && originalNullableEnumerator.MoveNext())
                {
                    Assert.True(CompareThrougReflection(enumeratorNullable.Current, originalNullableEnumerator.Current));
                }
                Assert.True(!enumeratorNullable.MoveNext() && !originalNullableEnumerator.MoveNext());
            }
        }

        public class ConversionNotSupportedMinValueClass
        {
            public int fInt;
            public long fLong;
            public short fShort;
            public sbyte fSByte;
        }

        [Fact]
        public void ConversionExceptionsBehavior()
        {
            using (var env = new TlcEnvironment())
            {
                var data = new ConversionNotSupportedMinValueClass[1];
                foreach (var field in typeof(ConversionNotSupportedMinValueClass).GetFields())
                {
                    data[0] = new ConversionNotSupportedMinValueClass();
                    bool gotException = false;
                    FieldInfo fi;
                    if ((fi = field.FieldType.GetField("MinValue")) != null)
                    {
                        field.SetValue(data[0], fi.GetValue(null));
                    }
                    var dataView = ComponentCreation.CreateDataView(env, data);
                    var enumerator = dataView.AsEnumerable<ConversionNotSupportedMinValueClass>(env, false).GetEnumerator();
                    try
                    {
                        enumerator.MoveNext();
                    }
                    catch
                    {
                        gotException = true;
                    }
                    Assert.True(gotException);
                }
            }
        }

        public class ConversionLossMinValueClass
        {
            public int? fInt;
            public long? fLong;
            public short? fShort;
            public sbyte? fSByte;
        }

        [Fact]
        public void ConversionMinValueToNullBehavior()
        {
            using (var env = new TlcEnvironment())
            {
                var data = new List<ConversionLossMinValueClass>(){
                    new ConversionLossMinValueClass(){ fSByte = null,fInt = null,fLong = null,fShort = null},
                    new ConversionLossMinValueClass(){fSByte = sbyte.MinValue,fInt = int.MinValue,fLong = long.MinValue,fShort = short.MinValue}
                };

                foreach (var field in typeof(ConversionLossMinValueClass).GetFields())
                {

                    var dataView = ComponentCreation.CreateDataView(env, data);
                    var enumerator = dataView.AsEnumerable<ConversionLossMinValueClass>(env, false).GetEnumerator();
                    while (enumerator.MoveNext())
                    {
                        Assert.True(enumerator.Current.fInt == null && enumerator.Current.fLong == null &&
                            enumerator.Current.fSByte == null && enumerator.Current.fShort == null);
                    }
                }
            }
        }

        [Fact]
        public void ClassWithConstFieldsConversion()
        {
            var data = new List<ClassWithConstField>()
            {
                new ClassWithConstField(){ fInt=1, fString ="lala" },
                new ClassWithConstField(){ fInt=-1, fString ="" },
                new ClassWithConstField(){ fInt=0, fString =null }
            };
            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var enumeratorSimple = dataView.AsEnumerable<ClassWithConstField>(env, false).GetEnumerator();

                var originalEnumerator = data.GetEnumerator();
                while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
                {
                    Assert.True(CompareThrougReflection(enumeratorSimple.Current, originalEnumerator.Current));
                }
                Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());
            }
        }

        public class ClassWithArrays
        {
            public string[] fString;
            public int[] fInt;
            public uint[] fuInt;
            public short[] fShort;
            public ushort[] fuShort;
            public sbyte[] fsByte;
            public byte[] fByte;
            public long[] fLong;
            public ulong[] fuLong;
            public float[] fFloat;
            public double[] fDouble;
            public bool[] fBool;
        }

        [Fact]
        public void BackAndForthConversionWithArrays()
        {
            var data = new List<ClassWithArrays>()
            {
                new ClassWithArrays(){ fInt = new int[3]{ 0,1,2}, fFloat = new float[3]{ -0.99f, 0f, 0.99f}, fString =new string[2]{ "hola", "lola"},
                    fBool =new bool[2]{true, false }, fByte = new byte[3]{ 0,124,255}, fDouble=new double[3]{ -1,0, 1}, fLong = new long[]{ 0,1,2} ,
                    fsByte = new sbyte[3]{ -127,127,0}, fShort = new short[3]{ 0, 1225, 32767 }, fuInt =new uint[2]{ 0, uint.MaxValue},
                    fuLong = new ulong[2]{ ulong.MaxValue, 0}, fuShort = new ushort[2]{ 0, ushort.MaxValue}
                },
                new ClassWithArrays(){ fInt = new int[3]{ -2,1,0}, fFloat = new float[3]{ 0.99f, 0f, -0.99f}, fString =new string[2]{  "lola","hola"} }
            };

            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var enumeratorSimple = dataView.AsEnumerable<ClassWithArrays>(env, false).GetEnumerator();
                var originalEnumerator = data.GetEnumerator();
                while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
                {
                    Assert.True(CompareThrougReflection(enumeratorSimple.Current, originalEnumerator.Current));
                }
                Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());
            }
        }
    }
}

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
                    var TextGetter = cursor.GetGetter<ReadOnlyMemory<char>>(1);

                    Assert.True(cursor.MoveNext());

                    float ID = 0;
                    IDGetter(ref ID);
                    Assert.Equal(1, ID);

                    ReadOnlyMemory<char> Text = new ReadOnlyMemory<char>();
                    TextGetter(ref Text);
                    Assert.Equal("1", Text.ToString());

                    Assert.True(cursor.MoveNext());

                    ID = 0;
                    IDGetter(ref ID);
                    Assert.Equal(2, ID);

                    Text = new ReadOnlyMemory<char>();
                    TextGetter(ref Text);
                    Assert.Equal("2", Text.ToString());

                    Assert.True(cursor.MoveNext());

                    ID = 0;
                    IDGetter(ref ID);
                    Assert.Equal(3, ID);

                    Text = new ReadOnlyMemory<char>();
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

        [Fact]
        public void CanTrainProperties()
        {
            var pipeline = new LearningPipeline();
            var data = new List<IrisDataProperties>() {
                new IrisDataProperties { SepalLength = 1f, SepalWidth = 1f, PetalLength=0.3f, PetalWidth=5.1f, Label=1},
                new IrisDataProperties { SepalLength = 1f, SepalWidth = 1f, PetalLength=0.3f, PetalWidth=5.1f, Label=1},
                new IrisDataProperties { SepalLength = 1.2f, SepalWidth = 0.5f, PetalLength=0.3f, PetalWidth=5.1f, Label=0}
            };
            var collection = CollectionDataSource.Create(data);

            pipeline.Add(collection);
            pipeline.Add(new ColumnConcatenator(outputColumn: "Features",
                "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            PredictionModel<IrisDataProperties, IrisPredictionProperties> model = pipeline.Train<IrisDataProperties, IrisPredictionProperties>();

            IrisPredictionProperties prediction = model.Predict(new IrisDataProperties()
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
            model = pipeline.Train<IrisDataProperties, IrisPredictionProperties>();

            prediction = model.Predict(new IrisDataProperties()
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

        public class IrisDataProperties
        {
            private float _Label;
            private float _SepalLength;
            private float _SepalWidth;
            private float _PetalLength;
            private float _PetalWidth;

            [Column("0")]
            public float Label { get { return _Label; } set { _Label = value; } }

            [Column("1")]
            public float SepalLength { get { return _SepalLength; } set { _SepalLength = value; } }

            [Column("2")]
            public float SepalWidth { get { return _SepalWidth; } set { _SepalWidth = value; } }

            [Column("3")]
            public float PetalLength { get { return _PetalLength; } set { _PetalLength = value; } }

            [Column("4")]
            public float PetalWidth { get { return _PetalWidth; } set { _PetalWidth = value; } }
        }

        public class IrisPredictionProperties
        {
            private float[] _PredictedLabels;
            [ColumnName("Score")]
            public float[] PredictedLabels { get { return _PredictedLabels; } set { _PredictedLabels = value; } }
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
            public string fString="";
        }

        public bool CompareObjectValues(object x, object y, Type type)
        {
            // By default behaviour for ReadOnlyMemory is to be empty string, while for string is null.
            // So if we do roundtrip string-> ReadOnlyMemory -> string all null string become empty strings.
            // Therefore replace all null values to empty string if field is string.
            if (type == typeof(string) && x == null)
                x = "";
            if (type == typeof(string) && y == null)
                y = "";
            if (x == null && y == null)
                return true;
            if (x == null && y != null)
                return false;
            return x.Equals(y);
        }

        public bool CompareThroughReflection<T>(T x, T y)
        {
            foreach (var field in typeof(T).GetFields(BindingFlags.Public | BindingFlags.Instance))
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
            foreach (var property in typeof(T).GetProperties(BindingFlags.Public | BindingFlags.Instance))
            {
                // Don't compare properties with private getters and setters
                if (!property.CanRead || !property.CanWrite || property.GetGetMethod() == null || property.GetSetMethod() == null)
                    continue;

                var xvalue = property.GetValue(x);
                var yvalue = property.GetValue(y);
                if (property.PropertyType.IsArray)
                {
                    if (!CompareArrayValues(xvalue as Array, yvalue as Array))
                        return false;
                }
                else
                {
                    if (!CompareObjectValues(xvalue, yvalue, property.PropertyType))
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

        [Fact]
        public void RoundTripConversionWithBasicTypes()
        {
            var data = new List<ConversionSimpleClass>
            {
                new ConversionSimpleClass()
                {
                    fInt = int.MaxValue - 1,
                    fuInt = uint.MaxValue - 1,
                    fBool = true,
                    fsByte = sbyte.MaxValue - 1,
                    fByte = byte.MaxValue - 1,
                    fDouble = double.MaxValue - 1,
                    fFloat = float.MaxValue - 1,
                    fLong = long.MaxValue - 1,
                    fuLong = ulong.MaxValue - 1,
                    fShort = short.MaxValue - 1,
                    fuShort = ushort.MaxValue - 1,
                    fString = ""
                },
                new ConversionSimpleClass()
                {
                    fInt = int.MaxValue,
                    fuInt = uint.MaxValue,
                    fBool = true,
                    fsByte = sbyte.MaxValue,
                    fByte = byte.MaxValue,
                    fDouble = double.MaxValue,
                    fFloat = float.MaxValue,
                    fLong = long.MaxValue,
                    fuLong = ulong.MaxValue,
                    fShort = short.MaxValue,
                    fuShort = ushort.MaxValue,
                    fString = "ooh"
                },
                new ConversionSimpleClass()
                {
                    fInt = int.MinValue + 1,
                    fuInt = uint.MinValue + 1,
                    fBool = false,
                    fsByte = sbyte.MinValue + 1,
                    fByte = byte.MinValue + 1,
                    fDouble = double.MinValue + 1,
                    fFloat = float.MinValue + 1,
                    fLong = long.MinValue + 1,
                    fuLong = ulong.MinValue + 1,
                    fShort = short.MinValue + 1,
                    fuShort = ushort.MinValue + 1,
                    fString = ""
                },
                new ConversionSimpleClass()
            };

            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var enumeratorSimple = dataView.AsEnumerable<ConversionSimpleClass>(env, false).GetEnumerator();
                var originalEnumerator = data.GetEnumerator();
                while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
                {
                    Assert.True(CompareThroughReflection(enumeratorSimple.Current, originalEnumerator.Current));
                }
                Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());
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
                        Assert.True(false);
                    }
                    catch
                    {
                    }
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

        public class ConversionLossMinValueClassProperties
        {
            private int? _fInt;
            private long? _fLong;
            private short? _fShort;
            private sbyte? _fsByte;
            public int? IntProp { get { return _fInt; } set { _fInt = value; } }
            public short? ShortProp { get { return _fShort; } set { _fShort = value; } }
            public sbyte? SByteProp { get { return _fsByte; } set { _fsByte = value; } }
            public long? LongProp { get { return _fLong; } set { _fLong = value; } }
        }

        public class ClassWithConstField
        {
            public const string ConstString = "N";
            public string fString;
            public const int ConstInt = 100;
            public int fInt;
        }

        [Fact]
        public void ClassWithConstFieldsConversion()
        {
            var data = new List<ClassWithConstField>()
            {
                new ClassWithConstField(){ fInt=1, fString ="lala" },
                new ClassWithConstField(){ fInt=-1, fString ="" },
            };

            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var enumeratorSimple = dataView.AsEnumerable<ClassWithConstField>(env, false).GetEnumerator();
                var originalEnumerator = data.GetEnumerator();
                while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
                    Assert.True(CompareThroughReflection(enumeratorSimple.Current, originalEnumerator.Current));
                Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());
            }
        }


        public class ClassWithMixOfFieldsAndProperties
        {
            public string fString;
            private int _fInt;
            public int IntProp { get { return _fInt; } set { _fInt = value; } }
        }

        [Fact]
        public void ClassWithMixOfFieldsAndPropertiesConversion()
        {
            var data = new List<ClassWithMixOfFieldsAndProperties>()
            {
                new ClassWithMixOfFieldsAndProperties(){ IntProp=1, fString ="lala" },
                new ClassWithMixOfFieldsAndProperties(){ IntProp=-1, fString ="" },
            };

            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var enumeratorSimple = dataView.AsEnumerable<ClassWithMixOfFieldsAndProperties>(env, false).GetEnumerator();
                var originalEnumerator = data.GetEnumerator();
                while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
                    Assert.True(CompareThroughReflection(enumeratorSimple.Current, originalEnumerator.Current));
                Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());
            }
        }

        public abstract class BaseClassWithInheritedProperties
        {
            private string _fString;
            private byte _fByte;
            public string StringProp { get { return _fString; } set { _fString = value; } }
            public abstract long LongProp { get; set; }
            public virtual byte ByteProp { get { return _fByte; } set { _fByte = value; } }
        }


        public class ClassWithPrivateFieldsAndProperties
        {
            public ClassWithPrivateFieldsAndProperties() { seq++; _unusedStaticField++; _unusedPrivateField1 = 100; }
            static public int seq;
            static public int _unusedStaticField;
            private int _unusedPrivateField1;
            private string _fString;

            [NoColumn]
            // This property can be used as source for DataView, but not casting from dataview to collection
            private int UnusedReadOnlyProperty { get { return _unusedPrivateField1; } }

            // This property is ignored because it is private 
            private int UnusedPrivateProperty { get { return _unusedPrivateField1; } set { _unusedPrivateField1 = value; } }

            [NoColumn]
            // This property can be used as source for DataView, but not casting from dataview to collection
            public int UnusedPropertyWithPrivateSetter { get { return _unusedPrivateField1; } private set { _unusedPrivateField1 = value; } }

            [NoColumn]
            // This property can be used as receptacle for dataview, but not as source for dataview.
            public int UnusedPropertyWithPrivateGetter { private get { return _unusedPrivateField1; } set { _unusedPrivateField1 = value; } }

            public string StringProp { get { return _fString; } set { _fString = value; } }
        }

        [Fact]
        public void ClassWithPrivateFieldsAndPropertiesConversion()
        {
            var data = new List<ClassWithPrivateFieldsAndProperties>()
            {
                new ClassWithPrivateFieldsAndProperties(){ StringProp ="lala" },
                new ClassWithPrivateFieldsAndProperties(){ StringProp ="baba" }
            };

            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var enumeratorSimple = dataView.AsEnumerable<ClassWithPrivateFieldsAndProperties>(env, false).GetEnumerator();
                var originalEnumerator = data.GetEnumerator();
                while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
                {
                    Assert.True(CompareThroughReflection(enumeratorSimple.Current, originalEnumerator.Current));
                    Assert.True(enumeratorSimple.Current.UnusedPropertyWithPrivateSetter == 100);
                }
                Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());
            }
        }

        public class ClassWithInheritedProperties : BaseClassWithInheritedProperties
        {
            private int _fInt;
            private long _fLong;
            private byte _fByte2;
            public int IntProp { get { return _fInt; } set { _fInt = value; } }
            public override long LongProp { get { return _fLong; } set { _fLong = value; } }
            public override byte ByteProp { get { return _fByte2; } set { _fByte2 = value; } }
        }

        [Fact]
        public void ClassWithInheritedPropertiesConversion()
        {
            var data = new List<ClassWithInheritedProperties>()
            {
                new ClassWithInheritedProperties(){ IntProp=1, StringProp ="lala", LongProp=17, ByteProp=3 },
                new ClassWithInheritedProperties(){ IntProp=-1, StringProp ="", LongProp=2, ByteProp=4 },
            };

            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var enumeratorSimple = dataView.AsEnumerable<ClassWithInheritedProperties>(env, false).GetEnumerator();
                var originalEnumerator = data.GetEnumerator();
                while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
                    Assert.True(CompareThroughReflection(enumeratorSimple.Current, originalEnumerator.Current));
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
        public void RoundTripConversionWithArrays()
        {

            var data = new List<ClassWithArrays>
            {
                new ClassWithArrays()
                {
                    fInt = new int[3] { 0, 1, 2 },
                    fFloat = new float[3] { -0.99f, 0f, 0.99f },
                    fString = new string[2] { "hola", "lola" },
                    fBool = new bool[2] { true, false },
                    fByte = new byte[3] { 0, 124, 255 },
                    fDouble = new double[3] { -1, 0, 1 },
                    fLong = new long[] { 0, 1, 2 },
                    fsByte = new sbyte[3] { -127, 127, 0 },
                    fShort = new short[3] { 0, 1225, 32767 },
                    fuInt = new uint[2] { 0, uint.MaxValue },
                    fuLong = new ulong[2] { ulong.MaxValue, 0 },
                    fuShort = new ushort[2] { 0, ushort.MaxValue }
                },
                new ClassWithArrays() { fInt = new int[3] { -2, 1, 0 }, fFloat = new float[3] { 0.99f, 0f, -0.99f }, fString = new string[2] { "", "" } },
                new ClassWithArrays()
            };

            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var enumeratorSimple = dataView.AsEnumerable<ClassWithArrays>(env, false).GetEnumerator();
                var originalEnumerator = data.GetEnumerator();
                while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
                {
                    Assert.True(CompareThroughReflection(enumeratorSimple.Current, originalEnumerator.Current));
                }
                Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());
            }
        }
        public class ClassWithArrayProperties
        {
            private string[] _fString;
            private int[] _fInt;
            private uint[] _fuInt;
            private short[] _fShort;
            private ushort[] _fuShort;
            private sbyte[] _fsByte;
            private byte[] _fByte;
            private long[] _fLong;
            private ulong[] _fuLong;
            private float[] _fFloat;
            private double[] _fDouble;
            private bool[] _fBool;
            public string[] StringProp { get { return _fString; } set { _fString = value; } }
            public int[] IntProp { get { return _fInt; } set { _fInt = value; } }
            public uint[] UIntProp { get { return _fuInt; } set { _fuInt = value; } }
            public short[] ShortProp { get { return _fShort; } set { _fShort = value; } }
            public ushort[] UShortProp { get { return _fuShort; } set { _fuShort = value; } }
            public sbyte[] SByteProp { get { return _fsByte; } set { _fsByte = value; } }
            public byte[] ByteProp { get { return _fByte; } set { _fByte = value; } }
            public long[] LongProp { get { return _fLong; } set { _fLong = value; } }
            public ulong[] ULongProp { get { return _fuLong; } set { _fuLong = value; } }
            public float[] FloatProp { get { return _fFloat; } set { _fFloat = value; } }
            public double[] DobuleProp { get { return _fDouble; } set { _fDouble = value; } }
            public bool[] BoolProp { get { return _fBool; } set { _fBool = value; } }
        }

        [Fact]
        public void RoundTripConversionWithArrayPropertiess()
        {

            var data = new List<ClassWithArrayProperties>
            {
                new ClassWithArrayProperties()
                {
                    IntProp = new int[3] { 0, 1, 2 },
                    FloatProp = new float[3] { -0.99f, 0f, 0.99f },
                    StringProp = new string[2] { "hola", "lola" },
                    BoolProp = new bool[2] { true, false },
                    ByteProp = new byte[3] { 0, 124, 255 },
                    DobuleProp = new double[3] { -1, 0, 1 },
                    LongProp = new long[] { 0, 1, 2 },
                    SByteProp = new sbyte[3] { -127, 127, 0 },
                    ShortProp = new short[3] { 0, 1225, 32767 },
                    UIntProp = new uint[2] { 0, uint.MaxValue },
                    ULongProp = new ulong[2] { ulong.MaxValue, 0 },
                    UShortProp = new ushort[2] { 0, ushort.MaxValue }
                },
                new ClassWithArrayProperties() { IntProp = new int[3] { -2, 1, 0 }, FloatProp = new float[3] { 0.99f, 0f, -0.99f }, StringProp = new string[2] { "", "Tom" } },
                new ClassWithArrayProperties()
            };

            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var enumeratorSimple = dataView.AsEnumerable<ClassWithArrayProperties>(env, false).GetEnumerator();
                var originalEnumerator = data.GetEnumerator();
                while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
                {
                    Assert.True(CompareThroughReflection(enumeratorSimple.Current, originalEnumerator.Current));
                }
                Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());
            }
        }

        class ClassWithGetter
        {
            private DateTime _dateTime = DateTime.Now;
            public float Day { get { return _dateTime.Day; } }
            public int Hour { get { return _dateTime.Hour; } }
        }

        class ClassWithSetter
        {
            public float Day { private get; set; }
            public int Hour { private get; set; }

            [NoColumn]
            public float GetDay => Day;
            [NoColumn]
            public int GetHour => Hour;
        }

        [Fact]
        public void PrivateGetSetProperties()
        {
            var data = new List<ClassWithGetter>()
            {
                new ClassWithGetter(),
                new ClassWithGetter(),
                new ClassWithGetter()
            };

            using (var env = new TlcEnvironment())
            {
                var dataView = ComponentCreation.CreateDataView(env, data);
                var enumeratorSimple = dataView.AsEnumerable<ClassWithSetter>(env, false).GetEnumerator();
                var originalEnumerator = data.GetEnumerator();
                while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
                {
                    Assert.True(enumeratorSimple.Current.GetDay == originalEnumerator.Current.Day &&
                        enumeratorSimple.Current.GetHour == originalEnumerator.Current.Hour);
                }
            }
        }
    }
}

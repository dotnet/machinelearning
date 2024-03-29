﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Data;
using Microsoft.ML.TestFramework;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.EntryPoints.Tests
{
    public class CollectionsDataViewTest : BaseTestClass
    {
        public CollectionsDataViewTest(ITestOutputHelper output)
            : base(output)
        {
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
            public string fString = "";
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
                    fString = null
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

            var env = new MLContext(1);
            var dataView = env.Data.LoadFromEnumerable(data);
            var enumeratorSimple = env.Data.CreateEnumerable<ConversionSimpleClass>(dataView, false).GetEnumerator();
            var originalEnumerator = data.GetEnumerator();
            while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
            {
                Assert.True(CompareThroughReflection(enumeratorSimple.Current, originalEnumerator.Current));
            }
            Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());
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
            var env = new MLContext(1);
            var data = new ConversionNotSupportedMinValueClass[1];
            foreach (var field in typeof(ConversionNotSupportedMinValueClass).GetFields())
            {
                data[0] = new ConversionNotSupportedMinValueClass();
                FieldInfo fi;
                if ((fi = field.FieldType.GetField("MinValue")) != null)
                {
                    field.SetValue(data[0], fi.GetValue(null));
                }
                var dataView = env.Data.LoadFromEnumerable(data);
                var enumerator = env.Data.CreateEnumerable<ConversionNotSupportedMinValueClass>(dataView, false).GetEnumerator();
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

            var env = new MLContext(1);
            var dataView = env.Data.LoadFromEnumerable(data);
            var enumeratorSimple = env.Data.CreateEnumerable<ClassWithConstField>(dataView, false).GetEnumerator();
            var originalEnumerator = data.GetEnumerator();
            while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
                Assert.True(CompareThroughReflection(enumeratorSimple.Current, originalEnumerator.Current));
            Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());
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

            var env = new MLContext(1);
            var dataView = env.Data.LoadFromEnumerable(data);
            var enumeratorSimple = env.Data.CreateEnumerable<ClassWithMixOfFieldsAndProperties>(dataView, false).GetEnumerator();
            var originalEnumerator = data.GetEnumerator();
            while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
                Assert.True(CompareThroughReflection(enumeratorSimple.Current, originalEnumerator.Current));
            Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());
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
            public static int seq;
            public static int _unusedStaticField;
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

            var env = new MLContext(1);
            var dataView = env.Data.LoadFromEnumerable(data);
            var enumeratorSimple = env.Data.CreateEnumerable<ClassWithPrivateFieldsAndProperties>(dataView, false).GetEnumerator();
            var originalEnumerator = data.GetEnumerator();
            while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
            {
                Assert.True(CompareThroughReflection(enumeratorSimple.Current, originalEnumerator.Current));
                Assert.True(enumeratorSimple.Current.UnusedPropertyWithPrivateSetter == 100);
            }
            Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());
        }

        public class ClassWithInheritedProperties : BaseClassWithInheritedProperties
        {
            private long _fLong;
            private byte _fByte2;
            public int IntProp { get; set; }
            public override long LongProp { get => _fLong; set => _fLong = value; }
            public override byte ByteProp { get => _fByte2; set => _fByte2 = value; }
        }

        [Fact]
        public void ClassWithInheritedPropertiesConversion()
        {
            var data = new List<ClassWithInheritedProperties>()
            {
                new ClassWithInheritedProperties(){ IntProp=1, StringProp ="lala", LongProp=17, ByteProp=3 },
                new ClassWithInheritedProperties(){ IntProp=-1, StringProp ="", LongProp=2, ByteProp=4 },
            };

            var env = new MLContext(1);
            var dataView = env.Data.LoadFromEnumerable(data);
            var enumeratorSimple = env.Data.CreateEnumerable<ClassWithInheritedProperties>(dataView, false).GetEnumerator();
            var originalEnumerator = data.GetEnumerator();
            while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
                Assert.True(CompareThroughReflection(enumeratorSimple.Current, originalEnumerator.Current));
            Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());
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

                    fByte = new byte[3] { 0, 124, 255 },
                    fDouble = new double[3] { -1, 0, 1 },
                    fLong = new long[] { 0, 1, 2 },
                    fsByte = new sbyte[3] { -127, 127, 0 },
                    fShort = new short[3] { 0, 1225, 32767 },
                    fuInt = new uint[2] { 0, uint.MaxValue },
                    fuLong = new ulong[2] { ulong.MaxValue, 0 },
                    fuShort = new ushort[2] { 0, ushort.MaxValue }
                },
                new ClassWithArrays() { fInt = new int[3] { -2, 1, 0 }, fFloat = new float[3] { 0.99f, 0f, -0.99f }, fString = new string[2] { "", null } },
                new ClassWithArrays()
            };


            var env = new MLContext(1);
            var dataView = env.Data.LoadFromEnumerable(data);
            var enumeratorSimple = env.Data.CreateEnumerable<ClassWithArrays>(dataView, false).GetEnumerator();
            var originalEnumerator = data.GetEnumerator();
            while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
            {
                Assert.True(CompareThroughReflection(enumeratorSimple.Current, originalEnumerator.Current));
            }
            Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());
        }
        public class ClassWithArrayProperties
        {
            public string[] StringProp { get; set; }
            public int[] IntProp { get; set; }
            public uint[] UIntProp { get; set; }
            public short[] ShortProp { get; set; }
            public ushort[] UShortProp { get; set; }
            public sbyte[] SByteProp { get; set; }
            public byte[] ByteProp { get; set; }
            public long[] LongProp { get; set; }
            public ulong[] ULongProp { get; set; }
            public float[] FloatProp { get; set; }
            public double[] DobuleProp { get; set; }
            public bool[] BoolProp { get; set; }
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
                new ClassWithArrayProperties() { IntProp = new int[3] { -2, 1, 0 }, FloatProp = new float[3] { 0.99f, 0f, -0.99f }, StringProp = new string[2] { "", null } },
                new ClassWithArrayProperties()
            };

            var env = new MLContext(1);
            var dataView = env.Data.LoadFromEnumerable(data);
            var enumeratorSimple = env.Data.CreateEnumerable<ClassWithArrayProperties>(dataView, false).GetEnumerator();
            var originalEnumerator = data.GetEnumerator();
            while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
                Assert.True(CompareThroughReflection(enumeratorSimple.Current, originalEnumerator.Current));
            Assert.True(!enumeratorSimple.MoveNext() && !originalEnumerator.MoveNext());
        }

        private sealed class ClassWithGetter
        {
            private DateTime _dateTime = DateTime.Now;
            public float Day => _dateTime.Day;
            public int Hour => _dateTime.Hour;
        }

        private sealed class ClassWithSetter
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

            var env = new MLContext(1);
            var dataView = env.Data.LoadFromEnumerable(data);
            var enumeratorSimple = env.Data.CreateEnumerable<ClassWithSetter>(dataView, false).GetEnumerator();
            var originalEnumerator = data.GetEnumerator();
            while (enumeratorSimple.MoveNext() && originalEnumerator.MoveNext())
            {
                Assert.True(enumeratorSimple.Current.GetDay == originalEnumerator.Current.Day &&
                    enumeratorSimple.Current.GetHour == originalEnumerator.Current.Hour);
            }
        }
    }
}

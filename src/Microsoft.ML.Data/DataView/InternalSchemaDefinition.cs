// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    using Conditional = System.Diagnostics.ConditionalAttribute;
    /// <summary>
    /// An internal class that holds the (already validated) mapping between a custom type and an IDataView schema.
    /// </summary>
    [BestFriend]
    internal sealed class InternalSchemaDefinition
    {
        public readonly Column[] Columns;

        public class Column
        {
            public readonly string ColumnName;
            public readonly MemberInfo MemberInfo;
            public readonly ParameterInfo ReturnParameterInfo;
            public readonly DataViewType ColumnType;
            public readonly bool IsComputed;
            public readonly Delegate Generator;
            private readonly Dictionary<string, AnnotationInfo> _annotations;
            public Dictionary<string, AnnotationInfo> Annotations { get { return _annotations; } }
            public Type ComputedReturnType { get { return ReturnParameterInfo.ParameterType.GetElementType(); } }
            public Type FieldOrPropertyType => (MemberInfo is FieldInfo) ? (MemberInfo as FieldInfo).FieldType : (MemberInfo as PropertyInfo).PropertyType;
            public Type OutputType => IsComputed ? ComputedReturnType : FieldOrPropertyType;

            public Column(string columnName, DataViewType columnType, MemberInfo memberInfo) :
                this(columnName, columnType, memberInfo, null, null)
            { }

            public Column(string columnName, DataViewType columnType, MemberInfo memberInfo,
                Dictionary<string, AnnotationInfo> metadataInfos) :
                this(columnName, columnType, memberInfo, null, metadataInfos)
            { }

            public Column(string columnName, DataViewType columnType, Delegate generator) :
                this(columnName, columnType, null, generator, null)
            { }

            public Column(string columnName, DataViewType columnType, Delegate generator,
                Dictionary<string, AnnotationInfo> metadataInfos) :
                this(columnName, columnType, null, generator, metadataInfos)
            { }

            private Column(string columnName, DataViewType columnType, MemberInfo memberInfo = null,
                Delegate generator = null, Dictionary<string, AnnotationInfo> metadataInfos = null)
            {
                Contracts.AssertNonEmpty(columnName);
                Contracts.AssertValue(columnType);
                Contracts.AssertValueOrNull(generator);

                if (generator == null)
                {
                    Contracts.AssertValue(memberInfo);
                    MemberInfo = memberInfo;
                }
                else
                {
                    var returnParameterInfo = generator.GetMethodInfo().GetParameters()[2];
                    Contracts.AssertValue(returnParameterInfo);
                    ReturnParameterInfo = returnParameterInfo;
                }

                ColumnName = columnName;
                ColumnType = columnType;
                IsComputed = generator != null;
                Generator = generator;
                _annotations = metadataInfos == null ? new Dictionary<string, AnnotationInfo>()
                    : metadataInfos.ToDictionary(entry => entry.Key, entry => entry.Value);

                AssertRep();
            }

            /// <summary>
            /// Function that checks whether the InternalSchemaDefinition.Column is a valid one.
            /// To be valid, the Column must:
            ///     1. Have non-empty values for ColumnName and ColumnType
            ///     2. Have a non-empty value for FieldInfo iff it is a field column, else
            ///        ReturnParameterInfo and Generator iff it is a computed column
            ///     3. Generator must have the method inputs (TRow rowObject,
            ///        long position, ref TValue outputValue) in that order.
            ///  </summary>
            [Conditional("DEBUG")]
            public void AssertRep()
            {
                // Check that all fields have values.
                Contracts.AssertNonEmpty(ColumnName);
                Contracts.AssertValue(ColumnType);
                Contracts.AssertValueOrNull(Generator);

                // If Column is computed type, it must have a generator.
                Contracts.Assert(IsComputed == (Generator != null));

                // Column must have either a generator or a memberInfo value.
                Contracts.Assert((Generator == null) != (MemberInfo == null));

                // Additional Checks if there is a generator.
                if (Generator == null)
                    return;
                Contracts.AssertValue(ReturnParameterInfo);

                // Checks input parameters are (someClass, long, ref value) in that order.
                var parameterInfos = Generator.GetMethodInfo().GetParameters().ToArray();
                var parameterTypes = (from pInfo in parameterInfos select pInfo.ParameterType).ToArray();
                Contracts.Assert(parameterTypes.Length == 3);
                Contracts.Assert(parameterTypes[2].IsByRef);
                Contracts.Assert(parameterTypes[1] == typeof(long));
                Contracts.Assert(!(parameterTypes[0].GetTypeInfo().IsPrimitive || parameterTypes[0] == typeof(string)));

                // Check that generator returns void.
                Contracts.Assert(Generator.GetMethodInfo().ReturnType == typeof(void));

                // Checks that the return type of the generator is compatible with ColumnType.
                GetVectorAndItemType(ComputedReturnType, "return type", out bool isVector, out Type itemType);
                Contracts.Assert(isVector == ColumnType is VectorType);
                Contracts.Assert(itemType == ColumnType.GetItemType().RawType);
            }
        }

        private InternalSchemaDefinition(Column[] columns)
        {
            Contracts.AssertValue(columns);
            Columns = columns;
        }

        /// <summary>
        /// Given a field or property info on a type, returns whether this appears to be a vector type,
        /// and also the associated data kind for this type. If a valid data type could not
        /// be determined, this will throw.
        /// </summary>
        /// <param name="memberInfo">The field or property info to inspect.</param>
        /// <param name="isVector">Whether this appears to be a vector type.</param>
        /// <param name="itemType">
        /// The corresponding <see cref="PrimitiveDataViewType"/> RawType of the type, or items of this type if vector.
        /// </param>
        public static void GetVectorAndItemType(MemberInfo memberInfo, out bool isVector, out Type itemType)
        {
            Contracts.AssertValue(memberInfo);
            switch (memberInfo)
            {
                case FieldInfo fieldInfo:
                    GetVectorAndItemType(fieldInfo.FieldType, fieldInfo.Name, out isVector, out itemType);
                    break;

                case PropertyInfo propertyInfo:
                    GetVectorAndItemType(propertyInfo.PropertyType, propertyInfo.Name, out isVector, out itemType);
                    break;

                default:
                    Contracts.Assert(false);
                    throw Contracts.ExceptNotSupp("Expected a FieldInfo or a PropertyInfo");
            }
        }

        /// <summary>
        /// Given a type and name for a variable, returns whether this appears to be a vector type,
        /// and also the associated data type for this type. If a valid data type could not
        /// be determined, this will throw.
        /// </summary>
        /// <param name="rawType">The type of the variable to inspect.</param>
        /// <param name="name">The name of the variable to inspect.</param>
        /// <param name="isVector">Whether this appears to be a vector type.</param>
        /// <param name="itemType">
        /// The corresponding <see cref="PrimitiveDataViewType"/> RawType of the type, or items of this type if vector.
        /// </param>
        public static void GetVectorAndItemType(Type rawType, string name, out bool isVector, out Type itemType)
        {
            // Determine whether this is a vector, and also determine the raw item type.
            isVector = true;
            if (rawType.IsArray)
                itemType = rawType.GetElementType();
            else if (rawType.IsGenericType && rawType.GetGenericTypeDefinition() == typeof(VBuffer<>))
                itemType = rawType.GetGenericArguments()[0];
            else
            {
                itemType = rawType;
                isVector = false;
            }

            if (itemType == typeof(string))
                itemType = typeof(ReadOnlyMemory<char>);
            else if (!itemType.TryGetDataKind(out _))
                throw Contracts.ExceptParam(nameof(rawType), "Could not determine an IDataView type for member {0}", name);
        }

        public static InternalSchemaDefinition Create(Type userType, SchemaDefinition.Direction direction)
        {
            var userSchemaDefinition = SchemaDefinition.Create(userType, direction);
            return Create(userType, userSchemaDefinition);
        }

        public static InternalSchemaDefinition Create(Type userType, SchemaDefinition userSchemaDefinition)
        {
            Contracts.AssertValue(userType);
            Contracts.AssertValueOrNull(userSchemaDefinition);

            if (userSchemaDefinition == null)
                userSchemaDefinition = SchemaDefinition.Create(userType);

            Column[] dstCols = new Column[userSchemaDefinition.Count];

            for (int i = 0; i < userSchemaDefinition.Count; ++i)
            {
                var col = userSchemaDefinition[i];
                if (col.MemberName == null)
                    throw Contracts.ExceptParam(nameof(userSchemaDefinition), "Null field name detected in schema definition");

                bool isVector;
                Type dataItemType;
                MemberInfo memberInfo = null;

                if (!col.IsComputed)
                {
                    memberInfo = userType.GetField(col.MemberName);

                    if (memberInfo == null)
                        memberInfo = userType.GetProperty(col.MemberName);

                    if (memberInfo == null)
                        throw Contracts.ExceptParam(nameof(userSchemaDefinition), "No field or property with name '{0}' found in type '{1}'",
                            col.MemberName,
                            userType.FullName);

                    //Clause to handle the field that may be used to expose the cursor channel.
                    //This field does not need a column.
                    if ((memberInfo is FieldInfo && (memberInfo as FieldInfo).FieldType == typeof(IChannel)) ||
                        (memberInfo is PropertyInfo && (memberInfo as PropertyInfo).PropertyType == typeof(IChannel)))
                        continue;

                    GetVectorAndItemType(memberInfo, out isVector, out dataItemType);
                }
                else
                {
                    var parameterType = col.ReturnType;
                    if (parameterType == null)
                        throw Contracts.ExceptParam(nameof(userSchemaDefinition), "No return parameter found in computed column.");
                    GetVectorAndItemType(parameterType, "returnType", out isVector, out dataItemType);
                }
                // Infer the column name.
                var colName = string.IsNullOrEmpty(col.ColumnName) ? col.MemberName : col.ColumnName;
                // REVIEW: Because order is defined, we allow duplicate column names, since producing an IDataView
                // with duplicate column names is completely legal. Possible objection is that we should make it less
                // convenient to produce "hidden" columns, since this may not be of practical use to users.

                DataViewType colType;
                if (col.ColumnType == null)
                {
                    // Infer a type as best we can.
                    PrimitiveDataViewType itemType = ColumnTypeExtensions.PrimitiveTypeFromType(dataItemType);
                    colType = isVector ? new VectorType(itemType) : (DataViewType)itemType;
                }
                else
                {
                    // Make sure that the types are compatible with the declared type, including
                    // whether it is a vector type.
                    VectorType columnVectorType = col.ColumnType as VectorType;
                    if (isVector != (columnVectorType != null))
                    {
                        throw Contracts.ExceptParam(nameof(userSchemaDefinition), "Column '{0}' is supposed to be {1}, but type of associated field '{2}' is {3}",
                            colName, columnVectorType != null ? "vector" : "scalar", col.MemberName, isVector ? "vector" : "scalar");
                    }
                    DataViewType itemType = columnVectorType?.ItemType ?? col.ColumnType;
                    if (itemType.RawType != dataItemType)
                    {
                        throw Contracts.ExceptParam(nameof(userSchemaDefinition), "Column '{0}' is supposed to have item type {1}, but associated field has type {2}",
                            colName, itemType.RawType, dataItemType);
                    }
                    colType = col.ColumnType;
                }

                dstCols[i] = col.IsComputed ?
                    new Column(colName, colType, col.Generator, col.Annotations)
                    : new Column(colName, colType, memberInfo, col.Annotations);

            }
            return new InternalSchemaDefinition(dstCols);
        }
    }
}

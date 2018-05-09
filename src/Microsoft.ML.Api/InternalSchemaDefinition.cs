// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.Api
{
    using Conditional = System.Diagnostics.ConditionalAttribute;

    /// <summary>
    /// An internal class that holds the (already validated) mapping between a custom type and an IDataView schema.
    /// </summary>
    public sealed class InternalSchemaDefinition
    {
        public readonly Column[] Columns;

        public class Column
        {
            public readonly string ColumnName;
            public readonly FieldInfo FieldInfo;
            public readonly ParameterInfo ReturnParameterInfo;
            public readonly ColumnType ColumnType;
            public readonly bool IsComputed;
            public readonly Delegate Generator;
            private readonly Dictionary<string, MetadataInfo> _metadata;
            public Dictionary<string, MetadataInfo> Metadata { get { return _metadata; } }
            public Type ReturnType {get { return ReturnParameterInfo.ParameterType.GetElementType(); }}

            public Column(string columnName, ColumnType columnType, FieldInfo fieldInfo) :
                this(columnName, columnType, fieldInfo, null, null) { }

            public Column(string columnName, ColumnType columnType, FieldInfo fieldInfo,
                Dictionary<string, MetadataInfo> metadataInfos) :
                this(columnName, columnType, fieldInfo, null, metadataInfos) { }

            public Column(string columnName, ColumnType columnType, Delegate generator) :
                this(columnName, columnType, null, generator, null) { }

            public Column(string columnName, ColumnType columnType, Delegate generator,
                Dictionary<string, MetadataInfo> metadataInfos) :
                this(columnName, columnType, null, generator, metadataInfos) { }

            private Column(string columnName, ColumnType columnType, FieldInfo fieldInfo = null,
                Delegate generator = null, Dictionary<string, MetadataInfo> metadataInfos = null)
            {
                Contracts.AssertNonEmpty(columnName);
                Contracts.AssertValue(columnType);
                Contracts.AssertValueOrNull(generator);

                if (generator == null)
                {
                    Contracts.AssertValue(fieldInfo);
                    FieldInfo = fieldInfo;
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
                _metadata = metadataInfos == null ? new Dictionary<string, MetadataInfo>()
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

                // Column must have either a generator or a fieldInfo value.
                Contracts.Assert((Generator == null) != (FieldInfo == null));

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
                bool isVector;
                DataKind datakind;
                GetVectorAndKind(ReturnType, "return type", out isVector, out datakind);
                Contracts.Assert(isVector == ColumnType.IsVector);
                Contracts.Assert(datakind == ColumnType.ItemType.RawKind);
            }

        }

        private InternalSchemaDefinition(Column[] columns)
        {
            Contracts.AssertValue(columns);
            Columns = columns;
        }

        /// <summary>
        /// Given a field info on a type, returns whether this appears to be a vector type,
        /// and also the associated data kind for this type. If a data kind could not
        /// be determined, this will throw. 
        /// </summary>
        /// <param name="fieldInfo">The field info to inspect.</param>
        /// <param name="isVector">Whether this appears to be a vector type.</param>
        /// <param name="kind">The data kind of the type, or items of this type if vector.</param>
        public static void GetVectorAndKind(FieldInfo fieldInfo, out bool isVector, out DataKind kind)
        {
            Contracts.AssertValue(fieldInfo);
            Type rawFieldType = fieldInfo.FieldType;
            var name = fieldInfo.Name;
            GetVectorAndKind(rawFieldType, name, out isVector, out kind);
        }

        /// <summary>
        /// Given a parameter info on a type, returns whether this appears to be a vector type,
        /// and also the associated data kind for this type. If a data kind could not
        /// be determined, this will throw. 
        /// </summary>
        /// <param name="parameterInfo">The parameter info to inspect.</param>
        /// <param name="isVector">Whether this appears to be a vector type.</param>
        /// <param name="kind">The data kind of the type, or items of this type if vector.</param>
        public static void GetVectorAndKind(ParameterInfo parameterInfo, out bool isVector, out DataKind kind)
        {
            Contracts.AssertValue(parameterInfo);
            Type rawParameterType = parameterInfo.ParameterType;
            var name = parameterInfo.Name;
            GetVectorAndKind(rawParameterType, name, out isVector, out kind);
        }

        /// <summary>
        /// Given a type and name for a variable, returns whether this appears to be a vector type,
        /// and also the associated data kind for this type. If a data kind could not
        /// be determined, this will throw. 
        /// </summary>
        /// <param name="rawType">The type of the variable to inspect.</param>
        /// <param name="name">The name of the variable to inspect.</param>
        /// <param name="isVector">Whether this appears to be a vector type.</param>
        /// <param name="kind">The data kind of the type, or items of this type if vector.</param>
        public static void GetVectorAndKind(Type rawType, string name, out bool isVector, out DataKind kind)
        {
            // Determine whether this is a vector, and also determine the raw item type.
            Type rawItemType;
            isVector = true;
            if (rawType.IsArray)
                rawItemType = rawType.GetElementType();
            else if (rawType.IsGenericType && rawType.GetGenericTypeDefinition() == typeof(VBuffer<>))
                rawItemType = rawType.GetGenericArguments()[0];
            else
            {
                rawItemType = rawType;
                isVector = false;
            }

            // Get the data kind, and the item's column type.
            if (rawItemType == typeof(string))
                kind = DataKind.Text;
            else if (!rawItemType.TryGetDataKind(out kind))
                throw Contracts.ExceptParam(nameof(rawType), "Could not determine an IDataView type for member {0}", name);
        }

        public static InternalSchemaDefinition Create(Type userType, SchemaDefinition userSchemaDefinition = null)
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
                DataKind kind;
                FieldInfo fieldInfo = null;

                if (!col.IsComputed)
                {
                    fieldInfo = userType.GetField(col.MemberName);

                    if (fieldInfo == null)
                        throw Contracts.ExceptParam(nameof(userSchemaDefinition), "No field with name '{0}' found in type '{1}'",
                            col.MemberName,
                            userType.FullName);

                    //Clause to handle the field that may be used to expose the cursor channel. 
                    //This field does not need a column.
                    if (fieldInfo.FieldType == typeof(IChannel))
                        continue;

                    GetVectorAndKind(fieldInfo, out isVector, out kind);
                }
                else
                {
                    var parameterType = col.ReturnType;
                    if (parameterType == null)
                        throw Contracts.ExceptParam(nameof(userSchemaDefinition), "No return parameter found in computed column.");
                    GetVectorAndKind(parameterType, "returnType", out isVector, out kind);
                }
                // Infer the column name.
                var colName = string.IsNullOrEmpty(col.ColumnName) ? col.MemberName : col.ColumnName;
                // REVIEW: Because order is defined, we allow duplicate column names, since producing an IDataView
                // with duplicate column names is completely legal. Possible objection is that we should make it less
                // convenient to produce "hidden" columns, since this may not be of practical use to users.

                ColumnType colType;
                if (col.ColumnType == null)
                {
                    // Infer a type as best we can.
                    PrimitiveType itemType = PrimitiveType.FromKind(kind);
                    colType = isVector ? new VectorType(itemType) : (ColumnType)itemType;
                }
                else
                {
                    // Make sure that the types are compatible with the declared type, including 
                    // whether it is a vector type.
                    if (isVector != col.ColumnType.IsVector)
                    {
                        throw Contracts.ExceptParam(nameof(userSchemaDefinition), "Column '{0}' is supposed to be {1}, but type of associated field '{2}' is {3}",
                            colName, col.ColumnType.IsVector ? "vector" : "scalar", col.MemberName, isVector ? "vector" : "scalar");
                    }
                    if (kind != col.ColumnType.ItemType.RawKind)
                    {
                        throw Contracts.ExceptParam(nameof(userSchemaDefinition), "Column '{0}' is supposed to have item kind {1}, but associated field has kind {2}",
                            colName, col.ColumnType.ItemType.RawKind, kind);
                    }
                    colType = col.ColumnType;
                }

                dstCols[i] = col.IsComputed ?
                    new Column(colName, colType, col.Generator, col.Metadata)
                    : new Column(colName, colType, fieldInfo, col.Metadata);

            }
            return new InternalSchemaDefinition(dstCols);
        }
    }
}

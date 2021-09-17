// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Allow member to be marked as a <see cref="KeyDataViewType"/>.
    /// </summary>
    /// <remarks>
    /// Can be applied only for member of following types: <see cref="byte"/>, <see cref="ushort"/>, <see cref="uint"/>, <see cref="ulong"/>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class KeyTypeAttribute : Attribute
    {
        /// <summary>
        /// Marks member as <see cref="KeyDataViewType"/>.
        /// </summary>
        /// <remarks>
        /// Cardinality of <see cref="KeyDataViewType"/> would be maximum legal value of member type.
        /// </remarks>
        public KeyTypeAttribute()
        {
            throw Contracts.ExceptNotSupp("Using KeyType without the Count parameter is not supported");
        }

        /// <summary>
        /// Marks member as <see cref="KeyDataViewType"/> and specifies <see cref="KeyDataViewType"/> cardinality.
        /// In case of the attribute being used with int types, the <paramref name="count"/> should be set to one more than
        /// the maximum value to account for counting starting at 1 (0 is reserved for the missing KeyType). E.g the cardinality of the
        /// 0-9 range is 10.
        /// If the values are outside of the specified cardinality they will be mapped to the missing value representation: 0.
        /// </summary>
        /// <param name="count">Cardinality of <see cref="KeyDataViewType"/>.</param>
        public KeyTypeAttribute(ulong count)
        {
            KeyCount = new KeyCount(count);
        }

        /// <summary>
        /// The key count.
        /// </summary>
        internal KeyCount KeyCount { get; }
    }

    /// <summary>
    /// Allows a member to be marked as a <see cref="VectorDataViewType"/>, primarily allowing one to set
    /// the dimensionality of the resulting array.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class VectorTypeAttribute : Attribute
    {
        /// <summary>
        /// The length of the vectors from this vector valued field.
        /// </summary>
        internal int[] Dims { get; }

        /// <summary>
        /// Mark member as single-dimensional array with unknown size.
        /// </summary>
        public VectorTypeAttribute()
        {

        }
        /// <summary>
        /// Mark member as single-dimensional array with specified size.
        /// </summary>
        /// <param name="size">Expected size of array. A zero value indicates that the vector type is considered to have unknown length.</param>
        public VectorTypeAttribute(int size)
        {
            Contracts.CheckParam(size >= 0, nameof(size), "Should be non-negative number");
            Dims = new int[1] { size };
        }

        /// <summary>
        /// Mark member with expected dimensions of array. Notice that this attribute is expected to be added to one dimensional arrays,
        /// and it shouldn't be added to multidimensional arrays. Internally, ML.NET will use the shape information provided as the
        /// "dimensions" param of this constructor, to use it as a multidimensional array.
        /// </summary>
        /// <param name="dimensions">Dimensions of array. All values should be non-negative.
        /// A zero value indicates that the vector type is considered to have unknown length along that dimension.</param>
        public VectorTypeAttribute(params int[] dimensions)
        {
            foreach (var size in dimensions)
            {
                Contracts.CheckParam(size >= 0, nameof(dimensions), "Should contain only non-negative values");
            }
            Dims = dimensions;
        }
    }

    /// <summary>
    /// Allows a member to specify <see cref="IDataView"/> column name directly, as opposed to the default
    /// behavior of using the member name as the column name.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class ColumnNameAttribute : Attribute
    {
        /// <summary>
        /// Column name.
        /// </summary>
        internal string Name { get; }

        /// <summary>
        /// Allows one to specify a name to expose this column as, as opposed to the default
        /// behavior of using the member name as the column name.
        /// </summary>
        public ColumnNameAttribute(string name)
        {
            Name = name;
        }
    }

    /// <summary>
    /// Mark this member as not being exposed as a <see cref="IDataView"/> column in the <see cref="DataViewSchema"/>.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class NoColumnAttribute : Attribute
    {
    }

    /// <summary>
    /// Mark a member that implements exactly IChannel as being permitted to receive
    /// channel information from an external channel.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    [BestFriend]
    internal sealed class CursorChannelAttribute : Attribute
    {
        /// <summary>
        /// When passed some object, and a channel, it attempts to pass the channel to the object. It
        /// passes the channel to the object iff the object has exactly one field marked with the
        /// CursorChannelAttribute, and that field implements only the IChannel interface.
        ///
        /// The function returns the modified object, as well as a boolean indicator of whether it was
        /// able to pass the channel to the object.
        /// </summary>
        /// <param name="obj">The object that attempts to acquire the channel.</param>
        /// <param name="channel">The channel to pass to the object.</param>
        /// <param name="ectx">The exception context.</param>
        /// <returns>1. A boolean indicator of whether the channel was successfully passed to the object.
        /// 2. The object passed in (only modified by the addition of the channel to the field
        /// with the CursorChannelAttribute, if the channel was added successfully).</returns>
        public static bool TrySetCursorChannel<T>(IExceptionContext ectx, T obj, IChannel channel)
            where T : class
        {
            Contracts.AssertValueOrNull(ectx);
            ectx.AssertValue(obj);
            ectx.AssertValue(channel);

            //Get public non-static fields with the CursorChannelAttribute as an array.
            var cursorChannelAttrFields = typeof(T)
                .GetFields(BindingFlags.Public | BindingFlags.Instance)
                .Where(x => x.GetCustomAttributes(typeof(CursorChannelAttribute), false).Any())
                .ToArray();

            var cursorChannelAttrProperties = typeof(T)
                .GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Where(x => x.CanRead && x.CanWrite && x.GetGetMethod() != null && x.GetSetMethod() != null && x.GetIndexParameters().Length == 0)
                .Where(x => x.GetCustomAttributes(typeof(CursorChannelAttribute), false).Any());

            var cursorChannelAttrMembers = (cursorChannelAttrFields as IEnumerable<MemberInfo>).Concat(cursorChannelAttrProperties).ToArray();

            //Check that there is at most one such field.
            if (cursorChannelAttrMembers.Length == 0)
                return false;

            ectx.Check(cursorChannelAttrMembers.Length == 1,
                "Only one public field or property with CursorChannel attribute is allowed.");

            //Check that the marked field has type IChannel.
            var cursorChannelAttrMemberInfo = cursorChannelAttrMembers[0];
            switch (cursorChannelAttrMemberInfo)
            {
                case FieldInfo cursorChannelAttrFieldInfo:
                    ectx.Check(cursorChannelAttrFieldInfo.FieldType == typeof(IChannel),
                        "Field marked as CursorChannel must have type IChannel.");
                    cursorChannelAttrFieldInfo.SetValue(obj, channel);
                    break;

                case PropertyInfo cursorChannelAttrPropertyInfo:
                    ectx.Check(cursorChannelAttrPropertyInfo.PropertyType == typeof(IChannel),
                        "Property marked as CursorChannel must have type IChannel.");
                    cursorChannelAttrPropertyInfo.SetValue(obj, channel);
                    break;

                default:
                    Contracts.Assert(false);
                    throw Contracts.ExceptNotSupp("Expected a FieldInfo or a PropertyInfo");
            }
            return true;
        }
    }

    /// <summary>
    /// This class defines a schema of a typed data view.
    /// </summary>
    public sealed class SchemaDefinition : List<SchemaDefinition.Column>
    {
        /// <summary>
        /// One column of the data view.
        /// </summary>
        public sealed class Column
        {
            internal Dictionary<string, AnnotationInfo> AnnotationInfos { get; }

            /// <summary>
            /// The name of the member the column is taken from. The API
            /// requires this to not be null, and a valid name of a member of
            /// the type for which we are creating a schema.
            /// </summary>
            public string MemberName { get; }
            /// <summary>
            /// The name of the column that's created in the data view. If this
            /// is null, the API uses the <see cref="MemberName"/>.
            /// </summary>
            public string ColumnName { get; set; }
            /// <summary>
            /// The column type. If this is null, the API attempts to derive a type
            /// from the member's type.
            /// </summary>
            public DataViewType ColumnType { get; set; }

            /// <summary>
            /// The generator function. if the column is computed.
            /// </summary>
            internal Delegate Generator { get; set; }

            internal Type ReturnType => Generator?.GetMethodInfo().GetParameters().LastOrDefault().ParameterType.GetElementType();

            internal Column(string memberName, DataViewType columnType,
                string columnName = null)
            {
                Contracts.CheckNonEmpty(memberName, nameof(memberName));
                MemberName = memberName;
                ColumnName = columnName ?? memberName;
                ColumnType = columnType;
                AnnotationInfos = new Dictionary<string, AnnotationInfo>();
            }

            /// <summary>
            /// Add annotation to the column.
            /// </summary>
            /// <typeparam name="T">Type of annotation being added. Types sported as entries in columns
            /// are also supported as entries in Annotations. Multiple annotations may be added to one column.
            /// </typeparam>
            /// <param name="kind">The string identifier of the annotation.</param>
            /// <param name="value">Value of annotation.</param>
            /// <param name="annotationType">Type of value.</param>
            public void AddAnnotation<T>(string kind, T value, DataViewType annotationType)
            {
                Contracts.CheckValue(kind, nameof(kind));
                Contracts.CheckValue(annotationType, nameof(annotationType));

                if (AnnotationInfos.ContainsKey(kind))
                    throw Contracts.Except("Column already contains an annotation of this kind.");
                AnnotationInfos[kind] = new AnnotationInfo<T>(kind, value, annotationType);
            }

            internal void AddAnnotation(string kind, AnnotationInfo info)
            {
                AnnotationInfos[kind] = info;
            }

            /// <summary>
            /// Returns annotations kind and type associated with this column.
            /// </summary>
            /// <returns>A dictionary with the kind of the annotation as the key, and the
            /// annotation type as the associated value.</returns>
            public DataViewSchema.Annotations Annotations
            {
                get
                {
                    var builder = new DataViewSchema.Annotations.Builder();
                    foreach (var kvp in AnnotationInfos)
                        builder.Add(kvp.Key, kvp.Value.AnnotationType, kvp.Value.GetGetterDelegate());
                    return builder.ToAnnotations();
                }
            }
        }

        private SchemaDefinition()
        {
        }

        /// <summary>
        /// Get or set the column definition by column name.
        /// If there's no such column:
        /// - get returns null,
        /// - set adds a new column.
        /// If there's more than one column with the same name:
        /// - get returns the first column,
        /// - set replaces the first column.
        /// </summary>
        public Column this[string columnName]
        {
            get => this.FirstOrDefault(x => x.ColumnName == columnName);
            set
            {
                Contracts.CheckValue(value, nameof(value));
                if (value.ColumnName != columnName)
                {
                    throw Contracts.ExceptParam(nameof(columnName),
                        "The column name is specified as '{0}' but must match the selector '{1}'.",
                        value.ColumnName, columnName);
                }

                int index = FindIndex(x => x.ColumnName == columnName);
                if (index >= 0)
                    this[index] = value;
                else
                    Add(value);
            }
        }

        [Flags]
        public enum Direction
        {
            Read = 1,
            Write = 2,
            Both = Read | Write
        }

        internal static MemberInfo[] GetMemberInfos(Type userType, Direction direction)
        {
            // REVIEW: This will have to be updated whenever we start
            // supporting properties and not just fields.
            Contracts.CheckValue(userType, nameof(userType));

            var fieldInfos = userType.GetFields(BindingFlags.Public | BindingFlags.Instance);
            var propertyInfos =
                userType
                .GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Where(x => (((direction & Direction.Read) == Direction.Read && (x.CanRead && x.GetGetMethod() != null)) ||
                ((direction & Direction.Write) == Direction.Write && (x.CanWrite && x.GetSetMethod() != null))) &&
                x.GetIndexParameters().Length == 0);

            return (fieldInfos as IEnumerable<MemberInfo>).Concat(propertyInfos).ToArray();
        }

        internal static bool NeedToCheckMemberInfo(MemberInfo memberInfo)
        {
            switch (memberInfo)
            {
                // Clause to handle the field that may be used to expose the cursor channel.
                // This field does not need a column.
                // REVIEW: maybe validate the channel attribute now, instead
                // of later at cursor creation.
                case FieldInfo fieldInfo:
                    if (fieldInfo.FieldType == typeof(IChannel))
                        return false;

                    // Const fields do not need to be mapped.
                    if (fieldInfo.IsLiteral)
                        return false;

                    break;

                case PropertyInfo propertyInfo:
                    if (propertyInfo.PropertyType == typeof(IChannel))
                        return false;
                    break;

                default:
                    Contracts.Assert(false);
                    throw Contracts.ExceptNotSupp("Expected a FieldInfo or a PropertyInfo");
            }

            if (memberInfo.GetCustomAttribute<NoColumnAttribute>() != null)
                return false;

            return true;
        }

        internal static bool GetNameAndCustomAttributes(MemberInfo memberInfo, Type userType, HashSet<string> colNames, out string name, out IEnumerable<Attribute> customAttributes)
        {
            name = null;
            customAttributes = null;

            if (!NeedToCheckMemberInfo(memberInfo))
                return false;

            customAttributes = memberInfo.GetCustomAttributes();
            var customTypeAttributes = customAttributes.Where(x => x is DataViewTypeAttribute);
            if (customTypeAttributes.Count() > 1)
                throw Contracts.ExceptParam(nameof(userType), "Member {0} cannot be marked with multiple attributes, {1}, derived from {2}.",
                    memberInfo.Name, customTypeAttributes, typeof(DataViewTypeAttribute));
            else if (customTypeAttributes.Count() == 1)
            {
                var customTypeAttribute = (DataViewTypeAttribute)customTypeAttributes.First();
                customTypeAttribute.Register();
            }

            var mappingNameAttr = memberInfo.GetCustomAttribute<ColumnNameAttribute>();
            name = mappingNameAttr?.Name ?? memberInfo.Name;
            // Disallow duplicate names, because the field enumeration order is not actually
            // well defined, so we are not guaranteed to have consistent "hiding" from run to
            // run, across different .NET versions.
            if (!colNames.Add(name))
                throw Contracts.ExceptParam(nameof(userType), "Duplicate column name '{0}' detected, this is disallowed", name);

            return true;
        }

        /// <summary>
        /// Create a schema definition by enumerating all public fields of the given type.
        /// </summary>
        /// <param name="userType">The type to base the schema on.</param>
        /// <param name="direction">Accept fields and properties based on their direction.</param>
        /// <returns>The generated schema definition.</returns>
        public static SchemaDefinition Create(Type userType, Direction direction = Direction.Both)
        {
            var memberInfos = GetMemberInfos(userType, direction);

            SchemaDefinition cols = new SchemaDefinition();
            HashSet<string> colNames = new HashSet<string>();

            foreach (var memberInfo in memberInfos)
            {
                if (!GetNameAndCustomAttributes(memberInfo, userType, colNames, out string name, out IEnumerable<Attribute> customAttributes))
                    continue;

                InternalSchemaDefinition.GetVectorAndItemType(memberInfo, out bool isVector, out Type dataType);

                // Get the column type.
                DataViewType columnType;
                if (!DataViewTypeManager.Knows(dataType, customAttributes))
                {
                    PrimitiveDataViewType itemType;
                    var keyAttr = memberInfo.GetCustomAttribute<KeyTypeAttribute>();
                    if (keyAttr != null)
                    {
                        if (!KeyDataViewType.IsValidDataType(dataType))
                            throw Contracts.ExceptParam(nameof(userType), "Member {0} marked with KeyType attribute, but does not appear to be a valid kind of data for a key type", memberInfo.Name);
                        if (keyAttr.KeyCount == null)
                            itemType = new KeyDataViewType(dataType, dataType.ToMaxInt());
                        else
                            itemType = new KeyDataViewType(dataType, keyAttr.KeyCount.Count.GetValueOrDefault());
                    }
                    else
                        itemType = ColumnTypeExtensions.PrimitiveTypeFromType(dataType);

                    var vectorAttr = memberInfo.GetCustomAttribute<VectorTypeAttribute>();
                    if (vectorAttr != null && !isVector)
                        throw Contracts.ExceptParam(nameof(userType), $"Member {memberInfo.Name} marked with {nameof(VectorTypeAttribute)}, but does not appear to be a vector type", memberInfo.Name);
                    if (isVector)
                    {
                        int[] dims = vectorAttr?.Dims;
                        if (dims != null && dims.Any(d => d < 0))
                            throw Contracts.ExceptParam(nameof(userType), "Some of member {0}'s dimension lengths are negative");
                        if (Utils.Size(dims) == 0)
                            columnType = new VectorDataViewType(itemType, 0);
                        else
                            columnType = new VectorDataViewType(itemType, dims);
                    }
                    else
                        columnType = itemType;
                }
                else
                    columnType = DataViewTypeManager.GetDataViewType(dataType, customAttributes);

                cols.Add(new Column(memberInfo.Name, columnType, name));
            }

            return cols;
        }
    }
}

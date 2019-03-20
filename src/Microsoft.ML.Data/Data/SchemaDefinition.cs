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
    /// Allow member to be marked as a <see cref="KeyType"/>.
    /// </summary>
    /// <remarks>
    /// Can be applied only for member of following types: <see cref="byte"/>, <see cref="ushort"/>, <see cref="uint"/>, <see cref="ulong"/>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class KeyTypeAttribute : Attribute
    {
        /// <summary>
        /// Marks member as <see cref="KeyType"/>.
        /// </summary>
        /// <remarks>
        /// Cardinality of <see cref="KeyType"/> would be maximum legal value of member type.
        /// </remarks>
        public KeyTypeAttribute()
        {

        }

        /// <summary>
        /// Marks member as <see cref="KeyType"/> and specifies <see cref="KeyType"/> cardinality.
        /// </summary>
        /// <param name="count">Cardinality of <see cref="KeyType"/>.</param>
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
    /// Allows a member to be marked as a <see cref="VectorType"/>, primarily allowing one to set
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
        /// Mark member with expected dimensions of array.
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
        /// <returns>1. A boolean indicator of whether the channel was sucessfully passed to the object.
        /// 2. The object passed in (only modified by the addition of the channel to the field
        /// with the CursorChannelAttribute, if the channel was added sucessfully).</returns>
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
            private readonly Dictionary<string, AnnotationInfo> _annotations;
            internal Dictionary<string, AnnotationInfo> Annotations { get { return _annotations; } }

            /// <summary>
            /// The name of the member the column is taken from. The API
            /// requires this to not be null, and a valid name of a member of
            /// the type for which we are creating a schema.
            /// </summary>
            public string MemberName { get; set; }
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
            /// Whether the column is a computed type.
            /// </summary>
            public bool IsComputed { get { return Generator != null; } }

            /// <summary>
            /// The generator function. if the column is computed.
            /// </summary>
            public Delegate Generator { get; set; }

            public Type ReturnType => Generator?.GetMethodInfo().GetParameters().LastOrDefault().ParameterType.GetElementType();

            public Column(IExceptionContext ectx, string memberName, DataViewType columnType,
                string columnName = null, IEnumerable<AnnotationInfo> annotationInfos = null, Delegate generator = null)
            {
                ectx.CheckNonEmpty(memberName, nameof(memberName));
                MemberName = memberName;
                ColumnName = columnName ?? memberName;
                ColumnType = columnType;
                Generator = generator;
                _annotations = annotationInfos != null ?
                    annotationInfos.ToDictionary(m => m.Kind, m => m)
                    : new Dictionary<string, AnnotationInfo>();
            }

            public Column()
            {
                _annotations = _annotations ?? new Dictionary<string, AnnotationInfo>();
            }

            /// <summary>
            /// Add annotation to the column.
            /// </summary>
            /// <typeparam name="T">Type of annotation being added. Types suported as entries in columns
            /// are also supported as entries in Annotations. Multiple annotations may be added to one column.
            /// </typeparam>
            /// <param name="kind">The string identifier of the annotation.</param>
            /// <param name="value">Value of annotation.</param>
            /// <param name="annotationType">Type of value.</param>
            public void AddAnnotation<T>(string kind, T value, DataViewType annotationType = null)
            {
                if (_annotations.ContainsKey(kind))
                    throw Contracts.Except("Column already contains an annotation of this kind.");
                _annotations[kind] = new AnnotationInfo<T>(kind, value, annotationType);
            }

            /// <summary>
            /// Remove annotation from the column if it exists.
            /// </summary>
            /// <param name="kind">The string identifier of the annotation.</param>
            public void RemoveAnnotation(string kind)
            {
                if (_annotations.ContainsKey(kind))
                    _annotations.Remove(kind);
                throw Contracts.Except("Column does not contain an annotation of kind: " + kind);
            }

            /// <summary>
            /// Returns annotations kind and type associated with this column.
            /// </summary>
            /// <returns>A dictionary with the kind of the annotation as the key, and the
            /// annotation type as the associated value.</returns>
            public IEnumerable<KeyValuePair<string, DataViewType>> GetAnnotationTypes
            {
                get
                {
                    return Annotations.Select(x => new KeyValuePair<string, DataViewType>(x.Key, x.Value.AnnotationType));
                }
            }
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

        /// <summary>
        /// Create a schema definition by enumerating all public fields of the given type.
        /// </summary>
        /// <param name="userType">The type to base the schema on.</param>
        /// <param name="direction">Accept fields and properties based on their direction.</param>
        /// <returns>The generated schema definition.</returns>
        public static SchemaDefinition Create(Type userType, Direction direction = Direction.Both)
        {
            // REVIEW: This will have to be updated whenever we start
            // supporting properties and not just fields.
            Contracts.CheckValue(userType, nameof(userType));

            SchemaDefinition cols = new SchemaDefinition();
            HashSet<string> colNames = new HashSet<string>();

            var fieldInfos = userType.GetFields(BindingFlags.Public | BindingFlags.Instance);
            var propertyInfos =
                userType
                .GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Where(x => (((direction & Direction.Read) == Direction.Read && (x.CanRead && x.GetGetMethod() != null)) ||
                ((direction & Direction.Write) == Direction.Write && (x.CanWrite && x.GetSetMethod() != null))) &&
                x.GetIndexParameters().Length == 0);

            var memberInfos = (fieldInfos as IEnumerable<MemberInfo>).Concat(propertyInfos).ToArray();

            foreach (var memberInfo in memberInfos)
            {
                // Clause to handle the field that may be used to expose the cursor channel.
                // This field does not need a column.
                // REVIEW: maybe validate the channel attribute now, instead
                // of later at cursor creation.
                switch (memberInfo)
                {
                    case FieldInfo fieldInfo:
                        if (fieldInfo.FieldType == typeof(IChannel))
                            continue;

                        // Const fields do not need to be mapped.
                        if (fieldInfo.IsLiteral)
                            continue;

                        break;

                    case PropertyInfo propertyInfo:
                        if (propertyInfo.PropertyType == typeof(IChannel))
                            continue;
                        break;

                    default:
                        Contracts.Assert(false);
                        throw Contracts.ExceptNotSupp("Expected a FieldInfo or a PropertyInfo");
                }

                if (memberInfo.GetCustomAttribute<NoColumnAttribute>() != null)
                    continue;

                var mappingNameAttr = memberInfo.GetCustomAttribute<ColumnNameAttribute>();
                string name = mappingNameAttr?.Name ?? memberInfo.Name;
                // Disallow duplicate names, because the field enumeration order is not actually
                // well defined, so we are not gauranteed to have consistent "hiding" from run to
                // run, across different .NET versions.
                if (!colNames.Add(name))
                    throw Contracts.ExceptParam(nameof(userType), "Duplicate column name '{0}' detected, this is disallowed", name);

                InternalSchemaDefinition.GetVectorAndItemType(memberInfo, out bool isVector, out Type dataType);

                PrimitiveDataViewType itemType;
                var keyAttr = memberInfo.GetCustomAttribute<KeyTypeAttribute>();
                if (keyAttr != null)
                {
                    if (!KeyType.IsValidDataType(dataType))
                        throw Contracts.ExceptParam(nameof(userType), "Member {0} marked with KeyType attribute, but does not appear to be a valid kind of data for a key type", memberInfo.Name);
                    if (keyAttr.KeyCount == null)
                        itemType = new KeyType(dataType, dataType.ToMaxInt());
                    else
                        itemType = new KeyType(dataType, keyAttr.KeyCount.Count.GetValueOrDefault());
                }
                else
                    itemType = ColumnTypeExtensions.PrimitiveTypeFromType(dataType);

                // Get the column type.
                DataViewType columnType;
                var vectorAttr = memberInfo.GetCustomAttribute<VectorTypeAttribute>();
                if (vectorAttr != null && !isVector)
                    throw Contracts.ExceptParam(nameof(userType), $"Member {memberInfo.Name} marked with {nameof(VectorTypeAttribute)}, but does not appear to be a vector type", memberInfo.Name);
                if (isVector)
                {
                    int[] dims = vectorAttr?.Dims;
                    if (dims != null && dims.Any(d => d < 0))
                        throw Contracts.ExceptParam(nameof(userType), "Some of member {0}'s dimension lengths are negative");
                    if (Utils.Size(dims) == 0)
                        columnType = new VectorType(itemType, 0);
                    else
                        columnType = new VectorType(itemType, dims);
                }
                else
                    columnType = itemType;

                cols.Add(new Column() { MemberName = memberInfo.Name, ColumnName = name, ColumnType = columnType });
            }
            return cols;
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// Attach to a member of a class to indicate that the item type should be of class key.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class KeyTypeAttribute : Attribute
    {
        // REVIEW: Property based, but should I just have a constructor?

        /// <summary>
        /// The minimum key value.
        /// </summary>
        public ulong Min { get; set; }

        /// <summary>
        /// The key count, if it is a known cardinality key.
        /// </summary>
        public int Count { get; set; }

        /// <summary>
        /// Whether keys should be considered to be contiguous.
        /// </summary>
        public bool Contiguous { get; set; }
        /// <summary>
        /// Public KeyTypeAttribute constuctor.
        /// </summary>
        public KeyTypeAttribute()
        {
            Contiguous = true;
        }
    }

    /// <summary>
    /// Allows a member to be marked as a vector valued field, primarily allowing one to set
    /// the dimensionality of the resulting array.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class VectorTypeAttribute : Attribute
    {
        private readonly int[] _dims;

        /// <summary>
        /// The length of the vectors from this vector valued field.
        /// </summary>
        public int[] Dims { get { return _dims; } }

        public VectorTypeAttribute(params int[] dims)
        {
            _dims = dims;
        }
    }

    /// <summary>
    /// Describes column information such as name and the source columns indicies that this
    /// column encapsulates.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class LoadColumnAttribute : Attribute
    {

        public LoadColumnAttribute(int ordinal, string name = null, bool loadAllOthers = false)
        {
            Start = ordinal.ToString();
            Sources = new List<TextLoader.Range>();
            var range = new TextLoader.Range(ordinal);
            range.AllOther = loadAllOthers;
            Sources.Add(range);
        }

        public LoadColumnAttribute(string start, string end = null, string name = null, int[] columnIndexes = null)
        {
            Name = name;
            Start = start;
            End = end;
            ColumnIndexes = columnIndexes;

            Sources = new List<TextLoader.Range>();

            bool hasEnd = int.TryParse(end, out int endIndex);
            var range = hasEnd ? new TextLoader.Range(int.Parse(start), endIndex) : new TextLoader.Range(int.Parse(start));
            Sources.Add(range);

            if (columnIndexes != null)
            {
                foreach (var col in columnIndexes)
                    Sources.Add(new TextLoader.Range(col));
            }
        }

        // REVIEW : AllOther seems to work only for a single column. Verify.
        public LoadColumnAttribute(string start, string end, string name = null, bool loadInverseRange = false)
        {
            Name = name;
            LoadInverseRange = loadInverseRange;
            Start = start;
            End = end;

            Sources = new List<TextLoader.Range>();
            var range = new TextLoader.Range(int.Parse(start), int.Parse(end));
            range.AllOther = loadInverseRange;
            Sources.Add(range);
        }

        public LoadColumnAttribute(int[] columnIndexes, string name = null)
        {
            Name = name;
            ColumnIndexes = columnIndexes;

            Sources = new List<TextLoader.Range>();
            foreach (var col in columnIndexes)
                Sources.Add(new TextLoader.Range(col));
        }

        internal List<TextLoader.Range> Sources;

        /// <summary>
        /// Column name.
        /// </summary>
        public string Name { get; }

        /// <summary>
        /// The optional start index for loading a contiguous range of columns, or the single index in the case
        /// of loading a single column.
        /// Either this parameters, or the <see cref="ColumnIndexes"/> should be specified.
        /// </summary>
        public string Start { get; }

        /// <summary>
        /// Optional field, used to set the dataset columns range end index when loading a range of columns.
        /// </summary>
        public string End { get; }

        /// <summary>
        /// Optional field used to specify the distinct indices of the dataset columns that need to be loaded, and mapped to this
        /// <see cref="TextLoader.Column"/>.
        /// </summary>
        public int[] ColumnIndexes { get; }

        /// <summary>
        /// If this is set to true, the columns defined in the range through either the <see cref="Start"/>, <see cref="End"/> or the
        /// <see cref="ColumnIndexes"/> will be excluded from loading, and all the other ones will loaded and mapped to the <see cref="TextLoader.Column"/>.
        /// </summary>
        public bool LoadInverseRange { get; }
    }

    /// <summary>
    /// Describes column information such as name and the source columns indicies that this
    /// column encapsulates.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class ColumnAttribute : Attribute
    {
        public ColumnAttribute(string ordinal, string name = null)
        {
            Name = name;
        }

        /// <summary>
        /// Column name.
        /// </summary>
        public string Name { get; }
    }

    /// <summary>
    /// Allows a member to specify its column name directly, as opposed to the default
    /// behavior of using the member name as the column name.
    /// </summary>
    [AttributeUsage(AttributeTargets.Field | AttributeTargets.Property, AllowMultiple = false, Inherited = true)]
    public sealed class ColumnNameAttribute : Attribute
    {
        private readonly string _name;
        /// <summary>
        /// Column name.
        /// </summary>
        public string Name { get { return _name; } }

        /// <summary>
        /// Allows one to specify a name to expose this column as, as opposed to simply
        /// the field name.
        /// </summary>
        public ColumnNameAttribute(string name)
        {
            _name = name;
        }
    }

    /// <summary>
    /// Mark this member as not being exposed as a column in the schema.
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
    public sealed class CursorChannelAttribute : Attribute
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
            private readonly Dictionary<string, MetadataInfo> _metadata;
            internal Dictionary<string, MetadataInfo> Metadata { get { return _metadata; } }

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
            public ColumnType ColumnType { get; set; }

            /// <summary>
            /// Whether the column is a computed type.
            /// </summary>
            public bool IsComputed { get { return Generator != null; } }

            /// <summary>
            /// The generator function. if the column is computed.
            /// </summary>
            public Delegate Generator { get; set; }

            public Type ReturnType => Generator?.GetMethodInfo().GetParameters().LastOrDefault().ParameterType.GetElementType();

            public Column(IExceptionContext ectx, string memberName, ColumnType columnType,
                string columnName = null, IEnumerable<MetadataInfo> metadataInfos = null, Delegate generator = null)
            {
                ectx.CheckNonEmpty(memberName, nameof(memberName));
                MemberName = memberName;
                ColumnName = columnName ?? memberName;
                ColumnType = columnType;
                Generator = generator;
                _metadata = metadataInfos != null ?
                    metadataInfos.ToDictionary(m => m.Kind, m => m)
                    : new Dictionary<string, MetadataInfo>();
            }

            public Column()
            {
                _metadata = _metadata ?? new Dictionary<string, MetadataInfo>();
            }

            /// <summary>
            /// Add metadata to the column.
            /// </summary>
            /// <typeparam name="T">Type of Metadata being added. Types suported as entries in columns
            /// are also supported as entries in Metadata. Multiple metadata may be added to one column.
            /// </typeparam>
            /// <param name="kind">The string identifier of the metadata.</param>
            /// <param name="value">Value of metadata.</param>
            /// <param name="metadataType">Type of value.</param>
            public void AddMetadata<T>(string kind, T value, ColumnType metadataType = null)
            {
                if (_metadata.ContainsKey(kind))
                    throw Contracts.Except("Column already contains metadata of this kind.");
                _metadata[kind] = new MetadataInfo<T>(kind, value, metadataType);
            }

            /// <summary>
            /// Remove metadata from the column if it exists.
            /// </summary>
            /// <param name="kind">The string identifier of the metadata. </param>
            public void RemoveMetadata(string kind)
            {
                if (_metadata.ContainsKey(kind))
                    _metadata.Remove(kind);
                throw Contracts.Except("Column does not contain metadata of kind: " + kind);
            }

            /// <summary>
            /// Returns metadata kind and type associated with this column.
            /// </summary>
            /// <returns>A dictionary with the kind of the metadata as the key, and the
            /// metadata type as the associated value.</returns>
            public IEnumerable<KeyValuePair<string, ColumnType>> GetMetadataTypes
            {
                get
                {
                    return Metadata.Select(x => new KeyValuePair<string, ColumnType>(x.Key, x.Value.MetadataType));
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

                var mappingAttr = memberInfo.GetCustomAttribute<ColumnAttribute>();
                var mappingNameAttr = memberInfo.GetCustomAttribute<ColumnNameAttribute>();
                string name = mappingAttr?.Name ?? mappingNameAttr?.Name ?? memberInfo.Name;
                // Disallow duplicate names, because the field enumeration order is not actually
                // well defined, so we are not gauranteed to have consistent "hiding" from run to
                // run, across different .NET versions.
                if (!colNames.Add(name))
                    throw Contracts.ExceptParam(nameof(userType), "Duplicate column name '{0}' detected, this is disallowed", name);

                InternalSchemaDefinition.GetVectorAndKind(memberInfo, out bool isVector, out DataKind kind);

                PrimitiveType itemType;
                var keyAttr = memberInfo.GetCustomAttribute<KeyTypeAttribute>();
                if (keyAttr != null)
                {
                    if (!KeyType.IsValidDataKind(kind))
                        throw Contracts.ExceptParam(nameof(userType), "Member {0} marked with KeyType attribute, but does not appear to be a valid kind of data for a key type", memberInfo.Name);
                    itemType = new KeyType(kind, keyAttr.Min, keyAttr.Count, keyAttr.Contiguous);
                }
                else
                    itemType = PrimitiveType.FromKind(kind);

                // Get the column type.
                ColumnType columnType;
                var vectorAttr = memberInfo.GetCustomAttribute<VectorTypeAttribute>();
                if (vectorAttr != null && !isVector)
                    throw Contracts.ExceptParam(nameof(userType), "Member {0} marked with VectorType attribute, but does not appear to be a vector type", memberInfo.Name);
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

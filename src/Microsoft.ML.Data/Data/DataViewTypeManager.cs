// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Internal.CpuMath.Core;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML.Data
{
    /// <summary>
    /// A singleton class for managing the map between ML.NET <see cref="DataViewType"/> and C# <see cref="Type"/>.
    /// To support custom column type in <see cref="IDataView"/>, the column's underlying type (e.g., a C# class's type)
    /// should be registered with a class derived from <see cref="DataViewType"/>.
    /// </summary>
    public static class DataViewTypeManager
    {
        /// <summary>
        /// Types have been used in ML.NET type systems. They can have multiple-to-one type mapping.
        /// For example, UInt32 and Key can be mapped to <see langword="uint"/>. This class enforces one-to-one mapping for all
        /// user-registered types.
        /// </summary>
        private static readonly HashSet<Type> _bannedRawTypes = new HashSet<Type>()
        {
            typeof(Boolean), typeof(SByte), typeof(Byte),
            typeof(Int16), typeof(UInt16), typeof(Int32), typeof(UInt32),
            typeof(Int64), typeof(UInt64), typeof(Single), typeof(Double),
            typeof(string), typeof(ReadOnlySpan<char>), typeof(ReadOnlyMemory<char>),
            typeof(VBuffer<>), typeof(Nullable<>), typeof(DateTime), typeof(DateTimeOffset),
            typeof(TimeSpan), typeof(DataViewRowId)
        };

        /// <summary>
        /// Mapping from a <see cref="Type"/> plus its <see cref="Attribute"/>s to a <see cref="DataViewType"/>.
        /// </summary>
        private static readonly Dictionary<TypeWithAttributes, DataViewType> _rawTypeToDataViewTypeMap = new Dictionary<TypeWithAttributes, DataViewType>();

        /// <summary>
        /// Mapping from a <see cref="DataViewType"/> to a <see cref="Type"/> plus its <see cref="Attribute"/>s.
        /// </summary>
        private static readonly Dictionary<DataViewType, TypeWithAttributes> _dataViewTypeToRawTypeMap = new Dictionary<DataViewType, TypeWithAttributes>();

        /// <summary>
        /// The lock that one should acquire if the state of <see cref="DataViewTypeManager"/> will be accessed or modified.
        /// </summary>
        private static readonly object _lock = new object();

        /// <summary>
        /// Returns the <see cref="DataViewType"/> registered for <paramref name="type"/> and its <paramref name="typeAttributes"/>.
        /// </summary>
        internal static DataViewType GetDataViewType(Type type, IEnumerable<Attribute> typeAttributes = null)
        {
            //Filter attributes as we only care about DataViewTypeAttribute
            DataViewTypeAttribute typeAttr = null;
            if (typeAttributes != null)
            {
                typeAttributes = typeAttributes.Where(attr => attr.GetType().IsSubclassOf(typeof(DataViewTypeAttribute)));
                if (typeAttributes.Count() > 1)
                {
                    throw Contracts.ExceptParam(nameof(type), "Type {0} cannot be marked with multiple attributes, {1}, derived from {2}.",
                        type.Name, typeAttributes, typeof(DataViewTypeAttribute));
                }
                else if (typeAttributes.Count() == 1)
                {
                    typeAttr = typeAttributes.First() as DataViewTypeAttribute;
                }
            }
            lock (_lock)
            {
                // Compute the ID of type with extra attributes.
                var rawType = new TypeWithAttributes(type, typeAttr);

                // Get the DataViewType's ID which typeID is mapped into.
                if (!_rawTypeToDataViewTypeMap.TryGetValue(rawType, out DataViewType dataViewType))
                    throw Contracts.ExceptParam(nameof(type), $"The raw type {type} with attributes {typeAttributes} is not registered with a DataView type.");

                // Retrieve the actual DataViewType identified by dataViewType.
                return dataViewType;
            }
        }

        /// <summary>
        /// If <paramref name="type"/> has been registered with a <see cref="DataViewType"/>, this function returns <see langword="true"/>.
        /// Otherwise, this function returns <see langword="false"/>.
        /// </summary>
        internal static bool Knows(Type type, IEnumerable<Attribute> typeAttributes = null)
        {
            //Filter attributes as we only care about DataViewTypeAttribute
            DataViewTypeAttribute typeAttr = null;
            if (typeAttributes != null)
            {
                typeAttributes = typeAttributes.Where(attr => attr.GetType().IsSubclassOf(typeof(DataViewTypeAttribute)));
                if (typeAttributes.Count() > 1)
                {
                    throw Contracts.ExceptParam(nameof(type), "Type {0} cannot be marked with multiple attributes, {1}, derived from {2}.",
                        type.Name, typeAttributes, typeof(DataViewTypeAttribute));
                }
                else if (typeAttributes.Count() == 1)
                {
                    typeAttr = typeAttributes.First() as DataViewTypeAttribute;
                }
            }
            lock (_lock)
            {
                // Compute the ID of type with extra attributes.
                var rawType = new TypeWithAttributes(type, typeAttr);

                // Check if this ID has been associated with a DataViewType.
                // Note that the dictionary below contains (rawType, dataViewType) pairs (key type is TypeWithAttributes, and value type is DataViewType).
                if (_rawTypeToDataViewTypeMap.ContainsKey(rawType))
                    return true;
                else
                    return false;
            }
        }

        /// <summary>
        /// If <paramref name="dataViewType"/> has been registered with a <see cref="Type"/>, this function returns <see langword="true"/>.
        /// Otherwise, this function returns <see langword="false"/>.
        /// </summary>
        internal static bool Knows(DataViewType dataViewType)
        {
            lock (_lock)
            {
                // Check if this the ID has been associated with a DataViewType.
                // Note that the dictionary below contains (dataViewType, rawType) pairs (key type is DataViewType, and value type is TypeWithAttributes).
                if (_dataViewTypeToRawTypeMap.ContainsKey(dataViewType))
                    return true;
                else
                    return false;
            }
        }

        /// <summary>
        /// This function tells that <paramref name="dataViewType"/> should be representation of data in <paramref name="type"/> in
        /// ML.NET's type system. The registered <paramref name="type"/> must be a standard C# object's type.
        /// </summary>
        /// <param name="type">Native type in C#.</param>
        /// <param name="dataViewType">The corresponding type of <paramref name="type"/> in ML.NET's type system.</param>
        /// <param name="typeAttributes">The <see cref="Attribute"/>s attached to <paramref name="type"/>.</param>
        [Obsolete("This API is deprecated, please use the new form of Register which takes in a single DataViewTypeAttribute instead.", false)]
        public static void Register(DataViewType dataViewType, Type type, IEnumerable<Attribute> typeAttributes)
        {
            DataViewTypeAttribute typeAttr = null;
            if (typeAttributes != null)
            {
                if (typeAttributes.Count() > 1)
                {
                    throw Contracts.ExceptParam(nameof(type), $"Type {type} has too many attributes.");
                }
                else if (typeAttributes.Count() == 1)
                {
                    var attr = typeAttributes.First();
                    if (!attr.GetType().IsSubclassOf(typeof(DataViewTypeAttribute)))
                    {
                        throw Contracts.ExceptParam(nameof(type), $"Type {type} has an attribute that is not of DataViewTypeAttribute.");
                    }
                    else
                    {
                        typeAttr = attr as DataViewTypeAttribute;
                    }
                }
            }
            Register(dataViewType, type, typeAttr);
        }
        /// <summary>
        /// This function tells that <paramref name="dataViewType"/> should be representation of data in <paramref name="type"/> in
        /// ML.NET's type system. The registered <paramref name="type"/> must be a standard C# object's type.
        /// </summary>
        /// <param name="type">Native type in C#.</param>
        /// <param name="dataViewType">The corresponding type of <paramref name="type"/> in ML.NET's type system.</param>
        /// <param name="typeAttribute">The <see cref="DataViewTypeAttribute"/> attached to <paramref name="type"/>.</param>
        public static void Register(DataViewType dataViewType, Type type, DataViewTypeAttribute typeAttribute = null)
        {
            lock (_lock)
            {
                if (_bannedRawTypes.Contains(type))
                    throw Contracts.ExceptParam(nameof(type), $"Type {type} has been registered as ML.NET's default supported type, " +
                        $"so it can't not be registered again.");

                var rawType = new TypeWithAttributes(type, typeAttribute);

                if (_rawTypeToDataViewTypeMap.ContainsKey(rawType) && _rawTypeToDataViewTypeMap[rawType].Equals(dataViewType) &&
                    _dataViewTypeToRawTypeMap.ContainsKey(dataViewType) && _dataViewTypeToRawTypeMap[dataViewType].Equals(rawType))
                    // This type pair has been registered. Note that registering one data type pair multiple times is allowed.
                    return;

                if (_rawTypeToDataViewTypeMap.ContainsKey(rawType) && !_rawTypeToDataViewTypeMap[rawType].Equals(dataViewType))
                {
                    // There is a pair of (rawType, anotherDataViewType) in _typeToDataViewType so we cannot register
                    // (rawType, dataViewType) again. The assumption here is that one rawType can only be associated
                    // with one dataViewType.
                    var associatedDataViewType = _rawTypeToDataViewTypeMap[rawType];
                    throw Contracts.ExceptParam(nameof(type), $"Repeated type register. The raw type {type} " +
                        $"has been associated with {associatedDataViewType} so it cannot be associated with {dataViewType}.");
                }

                if (_dataViewTypeToRawTypeMap.ContainsKey(dataViewType) && !_dataViewTypeToRawTypeMap[dataViewType].Equals(rawType))
                {
                    // There is a pair of (dataViewType, anotherRawType) in _dataViewTypeToType so we cannot register
                    // (dataViewType, rawType) again. The assumption here is that one dataViewType can only be associated
                    // with one rawType.
                    var associatedRawType = _dataViewTypeToRawTypeMap[dataViewType].TargetType;
                    throw Contracts.ExceptParam(nameof(dataViewType), $"Repeated type register. The DataView type {dataViewType} " +
                        $"has been associated with {associatedRawType} so it cannot be associated with {type}.");
                }

                _rawTypeToDataViewTypeMap.Add(rawType, dataViewType);
                _dataViewTypeToRawTypeMap.Add(dataViewType, rawType);
            }
        }

        /// <summary>
        /// An instance of <see cref="TypeWithAttributes"/> represents an unique key of its <see cref="TargetType"/> and <see cref="_associatedAttribute"/>.
        /// </summary>
        private class TypeWithAttributes
        {
            /// <summary>
            /// The underlying type.
            /// </summary>
            public Type TargetType { get; }

            /// <summary>
            /// The underlying type's attributes. Together with <see cref="TargetType"/>, <see cref="_associatedAttribute"/> uniquely defines
            /// a key when using <see cref="TypeWithAttributes"/> as the key type in <see cref="Dictionary{TKey, TValue}"/>. Note that the
            /// uniqueness is determined by <see cref="Equals(object)"/> and <see cref="GetHashCode"/> below.
            /// </summary>
            private readonly DataViewTypeAttribute _associatedAttribute;

            public TypeWithAttributes(Type type, DataViewTypeAttribute attribute)
            {
                TargetType = type;
                _associatedAttribute = attribute;
            }

            public override bool Equals(object obj)
            {
                if (obj is TypeWithAttributes other)
                {
                    // Flag of having the same type.
                    var sameType = TargetType.Equals(other.TargetType);
                    // Flag of having the attribute configurations.
                    var sameAttributeConfig = true;

                    if (_associatedAttribute == null && other._associatedAttribute == null)
                        sameAttributeConfig = true;
                    else if (_associatedAttribute == null && other._associatedAttribute != null)
                        sameAttributeConfig = false;
                    else if (_associatedAttribute != null && other._associatedAttribute == null)
                        sameAttributeConfig = false;
                    else
                    {
                        sameAttributeConfig = _associatedAttribute.Equals(other._associatedAttribute);
                    }

                    return sameType && sameAttributeConfig;
                }
                return false;
            }

            /// <summary>
            /// This function computes a hashing ID from <see name="TargetType"/> and attributes attached to it.
            /// If a type is defined as a member in a <see langword="class"/>, <see name="Attributes"/> can be obtained by calling
            /// <see cref="MemberInfo.GetCustomAttributes(bool)"/>.
            /// </summary>
            public override int GetHashCode()
            {
                if (_associatedAttribute == null)
                    return TargetType.GetHashCode();

                var code = TargetType.GetHashCode();
                if (_associatedAttribute != null)
                {
                    code = Hashing.CombineHash(code, _associatedAttribute.GetHashCode());
                }
                return code;
            }

        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Internal.DataView;

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
        private static HashSet<Type> _bannedRawTypes = new HashSet<Type>()
        {
            typeof(Boolean), typeof(SByte), typeof(Byte),
            typeof(Int16), typeof(UInt16), typeof(Int32), typeof(UInt32),
            typeof(Int64), typeof(UInt64), typeof(Single), typeof(Double),
            typeof(string), typeof(ReadOnlySpan<char>), typeof(ReadOnlyMemory<char>),
            typeof(VBuffer<>), typeof(Nullable<>), typeof(DateTime), typeof(DateTimeOffset),
            typeof(TimeSpan), typeof(DataViewRowId)
        };

        /// <summary>
        /// Mapping from hashing ID of a <see cref="Type"/> and its <see cref="Attribute"/>s to hashing ID of a <see cref="DataViewType"/>.
        /// </summary>
        private static Dictionary<TypeWithAttributesId, DataViewTypeId> _typeIdToDataViewTypeIdMap = new Dictionary<TypeWithAttributesId, DataViewTypeId>();

        /// <summary>
        /// Mapping from hashing ID of a <see cref="DataViewType"/> to hashing ID of a <see cref="Type"/> and its <see cref="Attribute"/>s.
        /// </summary>
        private static Dictionary<DataViewTypeId, TypeWithAttributesId> _dataViewTypeIdToTypeIdMap = new Dictionary<DataViewTypeId, TypeWithAttributesId>();

        /// <summary>
        /// The lock that one should acquire if the state of <see cref="DataViewTypeManager"/> will be accessed or modified.
        /// </summary>
        private static object _lock = new object();

        /// <summary>
        /// Returns the <see cref="DataViewType"/> registered for <paramref name="rawType"/> and its <paramref name="rawTypeAttributes"/>.
        /// </summary>
        public static DataViewType GetDataViewType(Type rawType, IEnumerable<Attribute> rawTypeAttributes = null)
        {
            lock (_lock)
            {
                // Compute the ID of type with extra attributes.
                var typeId = new TypeWithAttributesId(rawType, rawTypeAttributes);

                // Get the DataViewType's ID which typeID is mapped into.
                if (!_typeIdToDataViewTypeIdMap.TryGetValue(typeId, out DataViewTypeId dataViewTypeId))
                    throw Contracts.ExceptParam(nameof(rawType), $"The raw type {rawType} with attributes {rawTypeAttributes} is not registered with a DataView type.");

                // Retrieve the actual DataViewType identified by dataViewTypeId.
                return dataViewTypeId.TargetType;
            }
        }

        /// <summary>
        /// If <paramref name="rawType"/> has been registered with a <see cref="DataViewType"/>, this function returns <see langword="true"/>.
        /// Otherwise, this function returns <see langword="false"/>.
        /// </summary>
        public static bool Knows(Type rawType, IEnumerable<Attribute> rawTypeAttributes = null)
        {
            lock (_lock)
            {
                // Compute the ID of type with extra attributes.
                var typeId = new TypeWithAttributesId(rawType, rawTypeAttributes);

                // Check if this ID has been associated with a DataViewType.
                // Note that the dictionary below contains (typeId, type) pairs (key is typeId, and value is type).
                if (_typeIdToDataViewTypeIdMap.ContainsKey(typeId))
                    return true;
                else
                    return false;
            }
        }

        /// <summary>
        /// If <paramref name="dataViewType"/> has been registered with a <see cref="Type"/>, this function returns <see langword="true"/>.
        /// Otherwise, this function returns <see langword="false"/>.
        /// </summary>
        public static bool Knows(DataViewType dataViewType)
        {
            lock (_lock)
            {
                // Compute the ID of the input DataViewType.
                var dataViewTypeId = new DataViewTypeId(dataViewType);

                // Check if this the ID has been associated with a DataViewType.
                // Note that the dictionary below contains (dataViewTypeId, type) pairs (key is dataViewTypeId, and value is type).
                if (_dataViewTypeIdToTypeIdMap.ContainsKey(dataViewTypeId))
                    return true;
                else
                    return false;
            }
        }

        /// <summary>
        /// This function tells that <paramref name="dataViewType"/> should be representation of data in <paramref name="rawType"/> in
        /// ML.NET's type system. The registered <paramref name="rawType"/> must be a standard C# object's type.
        /// </summary>
        /// <param name="rawType">Native type in C#.</param>
        /// <param name="dataViewType">The corresponding type of <paramref name="rawType"/> in ML.NET's type system.</param>
        /// <param name="rawTypeAttributes">The <see cref="Attribute"/>s attached to <paramref name="rawType"/>.</param>
        public static void Register(DataViewType dataViewType, Type rawType, IEnumerable<Attribute> rawTypeAttributes = null)
        {
            lock (_lock)
            {
                if (_bannedRawTypes.Contains(rawType))
                    throw Contracts.ExceptParam(nameof(rawType), $"Type {rawType} has been registered as ML.NET's default supported type, " +
                        $"so it can't not be registered again.");

                var rawTypeId = new TypeWithAttributesId(rawType, rawTypeAttributes);
                var dataViewTypeId = new DataViewTypeId(dataViewType);

                if (_typeIdToDataViewTypeIdMap.ContainsKey(rawTypeId) && _typeIdToDataViewTypeIdMap[rawTypeId].Equals(dataViewTypeId) &&
                    _dataViewTypeIdToTypeIdMap.ContainsKey(dataViewTypeId) && _dataViewTypeIdToTypeIdMap[dataViewTypeId].Equals(rawTypeId))
                    // This type pair has been registered. Note that registering one data type pair multiple times is allowed.
                    return;

                if (_typeIdToDataViewTypeIdMap.ContainsKey(rawTypeId) && !_typeIdToDataViewTypeIdMap[rawTypeId].Equals(dataViewTypeId))
                {
                    // There is a pair of (rawTypeId, anotherDataViewTypeId) in _typeIdToDataViewTypeId so we cannot register
                    // (rawTypeId, dataViewTypeId) again. The assumption here is that one rawTypeId can only be associated
                    // with one dataViewTypeId.
                    var associatedDataViewType = _typeIdToDataViewTypeIdMap[rawTypeId].TargetType;
                    throw Contracts.ExceptParam(nameof(rawType), $"Repeated type register. The raw type {rawType} " +
                        $"has been associated with {associatedDataViewType} so it cannot be associated with {dataViewType}.");
                }

                if (_dataViewTypeIdToTypeIdMap.ContainsKey(dataViewTypeId) && !_dataViewTypeIdToTypeIdMap[dataViewTypeId].Equals(rawTypeId))
                {
                    // There is a pair of (dataViewTypeId, anotherRawTypeId) in _dataViewTypeIdToTypeId so we cannot register
                    // (dataViewTypeId, rawTypeId) again. The assumption here is that one dataViewTypeId can only be associated
                    // with one rawTypeId.
                    var associatedRawType = _dataViewTypeIdToTypeIdMap[dataViewTypeId].TargetType;
                    throw Contracts.ExceptParam(nameof(dataViewType), $"Repeated type register. The DataView type {dataViewType} " +
                        $"has been associated with {associatedRawType} so it cannot be associated with {rawType}.");
                }

                _typeIdToDataViewTypeIdMap.Add(rawTypeId, dataViewTypeId);
                _dataViewTypeIdToTypeIdMap.Add(dataViewTypeId, rawTypeId);
            }
        }

        /// <summary>
        /// An instance of <see cref="TypeWithAttributesId"/> represents an unique key of its <see cref="TargetType"/> and <see cref="_associatedAttributes"/>.
        /// </summary>
        private class TypeWithAttributesId
        {
            public Type TargetType { get; }
            private IEnumerable<Attribute> _associatedAttributes;

            public TypeWithAttributesId(Type rawType, IEnumerable<Attribute> attributes)
            {
                TargetType = rawType;
                _associatedAttributes = attributes;
            }

            public override bool Equals(object obj)
            {
                if (obj is TypeWithAttributesId other)
                {
                    // Flag of having the same type.
                    var sameType = TargetType.Equals(other.TargetType);
                    // Flag of having the attribute configurations.
                    var sameAttributeConfig = true;

                    if (_associatedAttributes == null && other._associatedAttributes == null)
                        sameAttributeConfig = true;
                    else if (_associatedAttributes == null && other._associatedAttributes != null)
                        sameAttributeConfig = false;
                    else if (_associatedAttributes != null && other._associatedAttributes == null)
                        sameAttributeConfig = false;
                    else if (_associatedAttributes.Count() != other._associatedAttributes.Count())
                        sameAttributeConfig = false;
                    else
                    {
                        var zipped = _associatedAttributes.Zip(other._associatedAttributes, (attr, otherAttr) => (attr, otherAttr));
                        foreach (var (attr, otherAttr) in zipped)
                        {
                            if (!attr.Equals(otherAttr))
                                sameAttributeConfig = false;
                        }
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
                if (_associatedAttributes == null)
                    return TargetType.GetHashCode();

                var code = TargetType.GetHashCode();
                foreach (var attr in _associatedAttributes)
                    code = Hashing.CombineHash(code, attr.GetHashCode());
                return code;
            }

        }

        /// <summary>
        /// An instance of <see cref="DataViewTypeId"/> represents an unique key of its <see cref="TargetType"/>.
        /// </summary>
        private class DataViewTypeId
        {
            public DataViewType TargetType { get; }

            public DataViewTypeId(DataViewType type)
            {
                TargetType = type;
            }

            public override bool Equals(object obj)
            {
                if (obj is DataViewTypeId other)
                    return TargetType.Equals(other.TargetType);

                return false;
            }

            public override int GetHashCode()
            {
                return TargetType.GetHashCode();
            }
        }
    }
}

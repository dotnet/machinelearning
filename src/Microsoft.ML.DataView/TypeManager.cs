// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
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
        // Types have been used in ML.NET type systems. They can have multiple-to-one type mapping.
        // For example, UInt32 and Key can be mapped to uint. This class enforces one-to-one mapping for all
        // user-registered types.
        private static HashSet<Type> _bannedRawTypes = new HashSet<Type>()
        {
            typeof(Boolean), typeof(SByte), typeof(Byte),
            typeof(Int16), typeof(UInt16), typeof(Int32), typeof(UInt32),
            typeof(Int64), typeof(UInt64), typeof(Single), typeof(Double),
            typeof(string), typeof(ReadOnlySpan<char>)
        };

        /// <summary>
        /// Mapping from ID to a <see cref="Type"/>. The ID is the ID of <see cref="Type"/> in ML.NET's type system.
        /// </summary>
        private static Dictionary<int, Type> _idToTypeMap = new Dictionary<int, Type>();

        /// <summary>
        /// Mapping from ID to a <see cref="DataViewType"/> instance. The ID is the ID of <see cref="DataViewType"/> instance in ML.NET's type system.
        /// </summary>
        private static Dictionary<int, DataViewType> _idToDataViewTypeMap = new Dictionary<int, DataViewType>();

        /// <summary>
        /// Mapping from hashing ID of a <see cref="Type"/> and its <see cref="Attribute"/>s to hashing ID of a <see cref="DataViewType"/>.
        /// </summary>
        private static Dictionary<int, int> _typeIdToDataViewTypeIdMap = new Dictionary<int, int>();

        /// <summary>
        /// Mapping from hashing ID of a <see cref="DataViewType"/> to hashing ID of a <see cref="Type"/> and its <see cref="Attribute"/>s.
        /// </summary>
        private static Dictionary<int, int> _dataViewTypeIdToTypeIdMap = new Dictionary<int, int>();

        private static object _lock = new object();

        /// <summary>
        /// This function computes a hashing ID from <paramref name="rawType"/> and attributes attached to it.
        /// If a type is defined as a member in a <see langword="class"/>, <paramref name="rawTypeAttributes"/> can be obtained by calling
        /// <see cref="MemberInfo.GetCustomAttributes(bool)"/>.
        /// </summary>
        /// <returns></returns>
        private static int ComputeHashCode(Type rawType, params Attribute[] rawTypeAttributes)
        {
            var code = rawType.GetHashCode();
            for (int i = 0; i < rawTypeAttributes.Length; ++i)
                code = Hashing.CombineHash(code, rawTypeAttributes[i].GetHashCode());
            return code;
        }

        /// <summary>
        /// This function hashes a <see cref="DataViewType"/> and its own hashing code together.
        /// </summary>
        private static int ComputeHashCode(DataViewType dataViewType) => Hashing.CombineHash(dataViewType.GetType().GetHashCode(), dataViewType.GetHashCode());

        /// <summary>
        /// Returns the <see cref="DataViewType"/> registered for <paramref name="rawType"/> and its <paramref name="rawTypeAttributes"/>.
        /// </summary>
        public static DataViewType GetDataViewType(Type rawType, params Attribute[] rawTypeAttributes)
        {
            // Overall flow:
            //   type (Type) + attrs ----> type ID ----------------> associated DataViewType's ID ----------------> DataViewType
            //                     (hashing)      (dictionary look-up)                           (dictionary look-up)
            lock (_lock)
            {
                // Compute the ID of type with extra attributes.
                var typeId = ComputeHashCode(rawType, rawTypeAttributes);

                // Get the DataViewType's ID which typeID is mapped into.
                if (!_typeIdToDataViewTypeIdMap.TryGetValue(typeId, out int dataViewTypeId))
                    throw Contracts.ExceptParam(nameof(rawType), $"The raw type {rawType} with attributes {rawTypeAttributes} is not registered with a DataView type.");

                // Retrieve the actual DataViewType identified by dataViewTypeId.
                return _idToDataViewTypeMap[dataViewTypeId];
            }
        }

        /// <summary>
        /// If <paramref name="rawType"/> has been registered with a <see cref="DataViewType"/>, this function returns <see langword="true"/>.
        /// Otherwise, this function returns <see langword="false"/>.
        /// </summary>
        public static bool Knows(Type rawType, params Attribute[] rawTypeAttributes)
        {
            lock (_lock)
            {
                // Compute the ID of type with extra attributes.
                var typeId = ComputeHashCode(rawType, rawTypeAttributes);

                // Check if this ID has been associated with a DataViewType.
                // Note that the dictionary below contains (typeId, type) pairs (key is typeId, and value is type).
                if (_idToTypeMap.ContainsKey(typeId))
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
                var dataViewTypeId = ComputeHashCode(dataViewType);

                // Check if this the ID has been associated with a DataViewType.
                // Note that the dictionary below contains (dataViewTypeId, type) pairs (key is dataViewTypeId, and value is type).
                if (_idToDataViewTypeMap.ContainsKey(dataViewTypeId))
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
        public static void Register(DataViewType dataViewType, Type rawType, params Attribute[] rawTypeAttributes)
        {
            lock (_lock)
            {
                if (_bannedRawTypes.Contains(rawType))
                    throw Contracts.ExceptParam(nameof(rawType), $"Type {rawType} has been registered as ML.NET's default supported type, " +
                        $"so it can't not be registered again.");

                int rawTypeId = ComputeHashCode(rawType, rawTypeAttributes);
                int dataViewTypeId = ComputeHashCode(dataViewType);

                if (_typeIdToDataViewTypeIdMap.ContainsKey(rawTypeId) && _typeIdToDataViewTypeIdMap[rawTypeId] == dataViewTypeId &&
                    _dataViewTypeIdToTypeIdMap.ContainsKey(dataViewTypeId) && _dataViewTypeIdToTypeIdMap[dataViewTypeId] == rawTypeId)
                    // This type pair has been registered. Note that registering one data type pair multiple times is allowed.
                    return;

                if (_typeIdToDataViewTypeIdMap.ContainsKey(rawTypeId) && _typeIdToDataViewTypeIdMap[rawTypeId] != dataViewTypeId)
                {
                    // There is a pair of (rawTypeId, anotherDataViewTypeId) in _typeIdToDataViewTypeId so we cannot register
                    // (rawTypeId, dataViewTypeId) again. The assumption here is that one rawTypeId can only be associated
                    // with one dataViewTypeId.
                    var associatedDataViewType = _idToDataViewTypeMap[_typeIdToDataViewTypeIdMap[rawTypeId]];
                    throw Contracts.ExceptParam(nameof(rawType), $"Repeated type register. The raw type {rawType} " +
                        $"has been associated with {associatedDataViewType} so it cannot be associated with {dataViewType}.");
                }

                if (_dataViewTypeIdToTypeIdMap.ContainsKey(dataViewTypeId) && _dataViewTypeIdToTypeIdMap[dataViewTypeId] != rawTypeId)
                {
                    // There is a pair of (dataViewTypeId, anotherRawTypeId) in _dataViewTypeIdToTypeId so we cannot register
                    // (dataViewTypeId, rawTypeId) again. The assumption here is that one dataViewTypeId can only be associated
                    // with one rawTypeId.
                    var associatedRawType = _idToTypeMap[_dataViewTypeIdToTypeIdMap[dataViewTypeId]];
                    throw Contracts.ExceptParam(nameof(dataViewType), $"Repeated type register. The DataView type {dataViewType} " +
                        $"has been associated with {associatedRawType} so it cannot be associated with {rawType}.");
                }

                _typeIdToDataViewTypeIdMap.Add(rawTypeId, dataViewTypeId);
                _dataViewTypeIdToTypeIdMap.Add(dataViewTypeId, rawTypeId);

                _idToDataViewTypeMap[dataViewTypeId] = dataViewType;
                _idToTypeMap[rawTypeId] = rawType;
            }
        }
    }
}

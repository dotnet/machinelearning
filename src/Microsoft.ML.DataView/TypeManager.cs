// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Threading;
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

        private static Dictionary<Type, DataViewType> _rawTypeToDataViewTypeMap = new Dictionary<Type, DataViewType>();
        private static Dictionary<DataViewType, Type> _dataViewTypeToRawTypeMap = new Dictionary<DataViewType, Type>();
        private static SpinLock _lock = new SpinLock();

        /// <summary>
        /// Returns the <see cref="DataViewType"/> registered for <paramref name="type"/>.
        /// </summary>
        public static DataViewType GetDataViewType(Type type)
        {
            bool ownLock = false;
            DataViewType dataViewType = null;
            try
            {
                _lock.Enter(ref ownLock);
                if (!_rawTypeToDataViewTypeMap.ContainsKey(type))
                    throw Contracts.ExceptParam(nameof(type), $"The raw type {type} is not registered with a DataView type.");
                dataViewType = _rawTypeToDataViewTypeMap[type];
            }
            finally
            {
                if (ownLock) _lock.Exit();
            }
            return dataViewType;
        }

        /// <summary>
        /// If <paramref name="type"/> has been registered with a <see cref="DataViewType"/>, this function returns <see langword="true"/>.
        /// Otherwise, this function returns <see langword="false"/>.
        /// </summary>
        public static bool Knows(Type type)
        {
            bool ownLock = false;
            bool answer = false;
            try
            {
                _lock.Enter(ref ownLock);
                if (_rawTypeToDataViewTypeMap.ContainsKey(type))
                    answer = true;
            }
            finally
            {
                if (ownLock) _lock.Exit();
            }
            return answer;
        }

        /// <summary>
        /// If <paramref name="type"/> has been registered with a <see cref="Type"/>, this function returns <see langword="true"/>.
        /// Otherwise, this function returns <see langword="false"/>.
        /// </summary>
        public static bool Knows(DataViewType type)
        {
            bool ownLock = false;
            bool answer = false;
            try
            {
                _lock.Enter(ref ownLock);
                if (_dataViewTypeToRawTypeMap.ContainsKey(type))
                    answer = true;
            }
            finally
            {
                if (ownLock) _lock.Exit();
            }
            return answer;
        }

        /// <summary>
        /// This function tells that <paramref name="dataViewType"/> should be representation of data in <paramref name="rawType"/> in
        /// ML.NET's type system. The registered <paramref name="rawType"/> must be a standard C# object's type.
        /// </summary>
        /// <param name="rawType">Native type in C#.</param>
        /// <param name="dataViewType">The corresponding type of <paramref name="rawType"/> in ML.NET's type system.</param>
        public static void Register(Type rawType, DataViewType dataViewType)
        {
            bool ownLock = false;

            try
            {
                _lock.Enter(ref ownLock);

                if (_bannedRawTypes.Contains(rawType))
                    throw Contracts.ExceptParam(nameof(rawType), $"Type {rawType} has been registered as ML.NET's default supported type, " +
                        $"so it can't not be registered again.");

                // Registering the same pair of (rawType, dataViewType) multiple times is ok. However, a raw type can be associated
                // with only one DataView type.
                if (_rawTypeToDataViewTypeMap.ContainsKey(rawType) && _rawTypeToDataViewTypeMap[rawType] != dataViewType)
                    throw Contracts.ExceptParam(nameof(rawType), $"Repeated type register. The raw type {rawType} " +
                        $"has been associated with {_rawTypeToDataViewTypeMap[rawType]} so it cannot be associated with {dataViewType}.");

                // Registering the same pair of (rawType, dataViewType) multiple times is ok. However, a DataView type can be associated
                // with only one raw type.
                if (_dataViewTypeToRawTypeMap.ContainsKey(dataViewType) && _dataViewTypeToRawTypeMap[dataViewType] != rawType)
                    throw Contracts.ExceptParam(nameof(dataViewType), $"Repeated type register. The DataView type {dataViewType} " +
                        $"has been associated with {_dataViewTypeToRawTypeMap[dataViewType]} so it cannot be associated with {rawType}.");

                if (!_rawTypeToDataViewTypeMap.ContainsKey(rawType))
                    _rawTypeToDataViewTypeMap.Add(rawType, dataViewType);

                if (!_dataViewTypeToRawTypeMap.ContainsKey(dataViewType))
                    _dataViewTypeToRawTypeMap.Add(dataViewType, rawType);
            }
            finally
            {
                if (ownLock) _lock.Exit();
            }
        }
    }
}

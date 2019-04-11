using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using Microsoft.ML.Internal.DataView;

namespace Microsoft.ML.Data
{
    public static class TypeManager
    {
        // Types have been used in ML.NET type systems. They can have multiple-to-one type mapping.
        // For example, UInt32 and Key can be mapped to uint. We enforce one-to-one mapping for all
        // user-registered types.
        private static HashSet<Type> _notAllowedRawTypes;
        private static ConcurrentDictionary<Type, DataViewType> _rawTypeToDataViewTypeMap;
        private static ConcurrentDictionary<DataViewType, Type> _dataViewTypeToRawTypeMap;

        /// <summary>
        /// Constructor to initialize type mappings.
        /// </summary>
        static TypeManager()
        {
            _notAllowedRawTypes = new HashSet<Type>() {
                typeof(Boolean), typeof(SByte), typeof(Byte),
                typeof(Int16), typeof(UInt16), typeof(Int32), typeof(UInt32),
                typeof(Int64), typeof(UInt64), typeof(string), typeof(ReadOnlySpan<char>)
            };
            _rawTypeToDataViewTypeMap = new ConcurrentDictionary<Type, DataViewType>();
            _dataViewTypeToRawTypeMap = new ConcurrentDictionary<DataViewType, Type>();
        }

        public static DataViewType GetDataViewTypeOrNull(Type type)
        {
            if (_rawTypeToDataViewTypeMap.ContainsKey(type))
                return _rawTypeToDataViewTypeMap[type];
            else
                return null;
        }

        public static Type GetRawTypeOrNull(DataViewType type)
        {
            if (_dataViewTypeToRawTypeMap.ContainsKey(type))
                return _dataViewTypeToRawTypeMap[type];
            else
                return null;
        }

        public static void Register(Type rawType, DataViewType dataViewType)
        {
            if (_notAllowedRawTypes.Contains(rawType))
                throw Contracts.ExceptParam(nameof(rawType), $"Type {rawType} has been registered as ML.NET's default type. " +
                    $"so it can't not be registered again.");
            if (_rawTypeToDataViewTypeMap.ContainsKey(rawType))
                throw Contracts.ExceptParam(nameof(rawType), $"Repeated type registration. The raw type {rawType} " +
                    $"has been associated with {_rawTypeToDataViewTypeMap[rawType]}.");
            _rawTypeToDataViewTypeMap[rawType] = dataViewType;
            _dataViewTypeToRawTypeMap[dataViewType] = rawType;
        }
    }
}

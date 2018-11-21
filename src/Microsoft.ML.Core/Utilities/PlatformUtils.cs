// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.ObjectModel;
using System.Reflection;

namespace Microsoft.ML.Runtime.Internal.Utilities
{
    /// <summary>
    /// Contains extension methods that aid in building cross platform.
    /// </summary>
    [BestFriend]
    internal static class PlatformUtils
    {
        public static ReadOnlyCollection<T> AsReadOnly<T>(this T[] items)
        {
            if (items == null)
                return null;
            return new ReadOnlyCollection<T>(items);
        }

        public static bool IsGenericEx(this Type type, Type typeDef)
        {
            Contracts.AssertValue(type);
            Contracts.AssertValue(typeDef);
            var info = type.GetTypeInfo();
            return info.IsGenericType && info.GetGenericTypeDefinition() == typeDef;
        }

        public static Type[] GetGenericTypeArgumentsEx(this Type type)
        {
            Contracts.AssertValue(type);
            var typeInfo = IntrospectionExtensions.GetTypeInfo(type);
            return typeInfo.IsGenericTypeDefinition
                                ? typeInfo.GenericTypeParameters
                                : typeInfo.GenericTypeArguments;
        }
    }
}

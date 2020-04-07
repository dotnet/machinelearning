﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#nullable enable

using System;
using System.Collections.Immutable;
using System.Reflection;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Internal.Utilities
{
    /// <summary>
    /// Represents the <see cref="MethodInfo"/> for a generic function corresponding to <see cref="Func{T1, T2, TResult}"/>,
    /// with the following characteristics:
    ///
    /// <list type="bullet">
    /// <item><description>One generic type argument.</description></item>
    /// <item><description>A return value of <typeparamref name="TResult"/>.</description></item>
    /// </list>
    /// </summary>
    /// <typeparam name="T1">The type of the first parameter of the method.</typeparam>
    /// <typeparam name="T2">The type of the second parameter of the method.</typeparam>
    /// <typeparam name="TResult">The type of the return value of the method.</typeparam>
    internal abstract class FuncMethodInfo1<T1, T2, TResult> : FuncMethodInfo<T1, T2, TResult>
    {
        private ImmutableDictionary<Type, MethodInfo> _instanceMethodInfo;

        private protected FuncMethodInfo1(MethodInfo methodInfo)
            : base(methodInfo)
        {
            _instanceMethodInfo = ImmutableDictionary<Type, MethodInfo>.Empty;

            Contracts.CheckParam(GenericMethodDefinition.GetGenericArguments().Length == 1, nameof(methodInfo),
                "Should have exactly one generic type parameter but does not");
        }

        public MethodInfo MakeGenericMethod(Type typeArg1)
        {
            return ImmutableInterlocked.GetOrAdd(
                ref _instanceMethodInfo,
                typeArg1,
                (typeArg, methodInfo) => methodInfo.MakeGenericMethod(typeArg),
                GenericMethodDefinition);
        }
    }
}

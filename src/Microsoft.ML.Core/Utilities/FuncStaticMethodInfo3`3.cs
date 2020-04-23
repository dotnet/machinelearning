// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#nullable enable

using System;
using System.Reflection;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Internal.Utilities
{
    /// <summary>
    /// Represents the <see cref="MethodInfo"/> for a generic function corresponding to <see cref="Func{T1, T2, TResult}"/>,
    /// with the following characteristics:
    ///
    /// <list type="bullet">
    /// <item><description>The method is static.</description></item>
    /// <item><description>Three generic type arguments.</description></item>
    /// <item><description>A return value of <typeparamref name="TResult"/>.</description></item>
    /// </list>
    /// </summary>
    /// <typeparam name="T1">The type of the first parameter of the method.</typeparam>
    /// <typeparam name="T2">The type of the second parameter of the method.</typeparam>
    /// <typeparam name="TResult">The type of the return value of the method.</typeparam>
    internal sealed class FuncStaticMethodInfo3<T1, T2, TResult> : FuncMethodInfo3<T1, T2, TResult>
    {
        public FuncStaticMethodInfo3(Func<T1, T2, TResult> function)
            : base(function.Method)
        {
            Contracts.CheckParam(GenericMethodDefinition.IsStatic, nameof(function), "Should be a static method");
        }
    }
}

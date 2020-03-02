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
    /// Represents the <see cref="MethodInfo"/> for a generic function corresponding to <see cref="Func{T, TResult}"/>,
    /// with the following characteristics:
    ///
    /// <list type="bullet">
    /// <item><description>The method is static.</description></item>
    /// <item><description>One generic type argument.</description></item>
    /// <item><description>A return value of <typeparamref name="TResult"/>.</description></item>
    /// </list>
    /// </summary>
    /// <typeparam name="T">The type of the parameter of the method.</typeparam>
    /// <typeparam name="TResult">The type of the return value of the method.</typeparam>
    internal sealed class FuncStaticMethodInfo1<T, TResult> : FuncMethodInfo1<T, TResult>
    {
        public FuncStaticMethodInfo1(Func<T, TResult> function)
            : base(function.Method)
        {
            Contracts.CheckParam(GenericMethodDefinition.IsStatic, nameof(function), "Should be a static method");
        }
    }
}

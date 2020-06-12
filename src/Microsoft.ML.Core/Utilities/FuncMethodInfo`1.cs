// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#nullable enable

using System.Reflection;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Internal.Utilities
{
    internal abstract class FuncMethodInfo<TResult>
    {
        private protected FuncMethodInfo(MethodInfo methodInfo)
        {
            Contracts.CheckValue(methodInfo, nameof(methodInfo));
            Contracts.CheckParam(methodInfo.IsGenericMethod, nameof(methodInfo), "Should be generic but is not");

            GenericMethodDefinition = methodInfo.GetGenericMethodDefinition();
            Contracts.CheckParam(typeof(TResult).IsAssignableFrom(GenericMethodDefinition.ReturnType), nameof(methodInfo), "Cannot be generic on return type");
        }

        protected MethodInfo GenericMethodDefinition { get; }
    }
}

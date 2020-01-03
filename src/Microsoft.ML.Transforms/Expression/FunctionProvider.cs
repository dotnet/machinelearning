// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Reflection;

namespace Microsoft.ML.Transforms
{
    public delegate void SignatureFunctionProvider();

    /// <summary>
    /// This interface enables extending the ExprTransform language with additional functions.
    /// </summary>
    public interface IFunctionProvider
    {
        /// <summary>
        /// The namespace for this provider. This should be a legal identifier in the expression language.
        /// Multiple providers may contribute to the same namespace.
        /// </summary>
        string NameSpace { get; }

        /// <summary>
        /// Returns an array of overloads for the given function name. This may return null instead of an
        /// empty array. The returned MethodInfos should  be public static methods that can be freely invoked
        /// by IL in a different assembly. They should also be "pure" functions - with the output only
        /// depending on the inputs and NOT on any global state.
        /// </summary>
        MethodInfo[] Lookup(string name);

        /// <summary>
        /// If the function's value can be determined by the given subset of its arguments, this should
        /// return the resulting value. Note that this should only be called if values is non-empty and
        /// contains at least one null. If all the arguments are non-null, then the MethodInfo will be
        /// invoked to produce the value.
        /// </summary>
        /// <param name="name">The name of the function.</param>
        /// <param name="meth">The MethodInfo provided by Lookup. When there are multiple overloads of
        /// a function with a given name, this can be used to determine which overload is being used.</param>
        /// <param name="values">The values of the input arguments, with null for the non-constant arguments. This should
        /// only be called if there is at least one null.</param>
        /// <returns>The constant value, when it can be determined; null otherwise.</returns>
        object ResolveToConstant(string name, MethodInfo meth, object[] values);
    }
}

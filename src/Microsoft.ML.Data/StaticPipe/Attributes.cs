// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.StaticPipe
{
    /// <summary>
    /// An indicator to the analyzer that this type parameter ought to be a valid schema-shape object (e.g., a pipeline-column, or
    /// value-tuples or some other containing type of such) as the return type. Note that this attribute is typically only used in
    /// situations where a user might be essentially declaring that type, as opposed to using an already established shape type.
    /// So: a method that merely takes an already existing typed instance would tend on the other hand to not use this type parameter.
    /// To give an example:
    /// <see cref="Estimator{TInShape, TOutShape, TTransformer}.Append{TNewOutShape}(Func{TOutShape, TNewOutShape})"/>
    /// has the parameter on the new output tuple shape.
    ///
    /// The cost to not specifying this on such an entry point is that the compile time type-checks on the shape parameters will
    /// no longer be enforced, which is suboptimal given that the purpose of the statically typed interfaces is to have compile-time
    /// checks. However, it is not disastrous since the runtime checks will still be in effect.
    ///
    /// User code may use this attribute on their types if they have generic type parameters that interface with this library.
    /// </summary>
    [AttributeUsage(AttributeTargets.GenericParameter)]
    public sealed class IsShapeAttribute : Attribute
    {
    }
}

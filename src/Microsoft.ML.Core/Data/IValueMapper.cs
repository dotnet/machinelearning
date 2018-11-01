// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Runtime.Data
{
    /// <summary>
    /// Delegate type to map/convert a value.
    /// </summary>
    public delegate void ValueMapper<TSrc, TDst>(in TSrc src, ref TDst dst);

    /// <summary>
    /// Delegate type to map/convert among three values, for example, one input with two
    /// outputs, or two inputs with one output.
    /// </summary>
    public delegate void ValueMapper<TVal1, TVal2, TVal3>(in TVal1 val1, ref TVal2 val2, ref TVal3 val3);

    /// <summary>
    /// Interface for mapping a single input value (of an indicated ColumnType) to
    /// an output value (of an indicated ColumnType). This interface is commonly implemented
    /// by predictors. Note that the input and output ColumnTypes determine the proper
    /// type arguments for GetMapper, but typically contain additional information like
    /// vector lengths.
    /// </summary>
    public interface IValueMapper
    {
        ColumnType InputType { get; }
        ColumnType OutputType { get; }

        /// <summary>
        /// Get a delegate used for mapping from input to output values. Note that the delegate
        /// should only be used on a single thread - it should NOT be assumed to be safe for concurrency.
        /// </summary>
        ValueMapper<TSrc, TDst> GetMapper<TSrc, TDst>();
    }

    /// <summary>
    /// Interface for mapping a single input value (of an indicated ColumnType) to an output value
    /// plus distribution value (of indicated ColumnTypes). This interface is commonly implemented
    /// by predictors. Note that the input, output, and distribution ColumnTypes determine the proper
    /// type arguments for GetMapper, but typically contain additional information like
    /// vector lengths.
    /// </summary>
    public interface IValueMapperDist : IValueMapper
    {
        ColumnType DistType { get; }

        /// <summary>
        /// Get a delegate used for mapping from input to output values. Note that the delegate
        /// should only be used on a single thread - it should NOT be assumed to be safe for concurrency.
        /// </summary>
        ValueMapper<TSrc, TDst, TDist> GetMapper<TSrc, TDst, TDist>();
    }
}
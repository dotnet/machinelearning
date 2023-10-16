
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from ColumnArithmetic.OperationEnums.tt. Do not modify directly
namespace Microsoft.Data.Analysis
{
    internal enum BinaryOperation
    {
        Add,
        Subtract,
        Multiply,
        Divide,
        Modulo,
        And,
        Or,
        Xor,
    }

    internal enum BinaryIntOperation
    {
        LeftShift,
        RightShift,
    }

    internal enum ComparisonOperation
    {
        ElementwiseEquals,
        ElementwiseNotEquals,
        ElementwiseGreaterThanOrEqual,
        ElementwiseLessThanOrEqual,
        ElementwiseGreaterThan,
        ElementwiseLessThan,
    }
}
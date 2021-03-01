
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Generated from PrimitiveColumnContainer.BinaryOperations.tt. Do not modify directly

namespace Microsoft.Data.Analysis
{
    internal partial class PrimitiveColumnContainer<T>
        where T : struct
    {
        public PrimitiveColumnContainer<T> Add(PrimitiveColumnContainer<T> right)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Add(this, right);
            return this;
        }

        public PrimitiveColumnContainer<T> Add(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Add(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> Subtract(PrimitiveColumnContainer<T> right)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Subtract(this, right);
            return this;
        }

        public PrimitiveColumnContainer<T> Subtract(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Subtract(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> Multiply(PrimitiveColumnContainer<T> right)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Multiply(this, right);
            return this;
        }

        public PrimitiveColumnContainer<T> Multiply(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Multiply(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> Divide(PrimitiveColumnContainer<T> right)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Divide(this, right);
            return this;
        }

        public PrimitiveColumnContainer<T> Divide(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Divide(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> Modulo(PrimitiveColumnContainer<T> right)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Modulo(this, right);
            return this;
        }

        public PrimitiveColumnContainer<T> Modulo(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Modulo(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> And(PrimitiveColumnContainer<T> right)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.And(this, right);
            return this;
        }

        public PrimitiveColumnContainer<T> And(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.And(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> Or(PrimitiveColumnContainer<T> right)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Or(this, right);
            return this;
        }

        public PrimitiveColumnContainer<T> Or(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Or(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> Xor(PrimitiveColumnContainer<T> right)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Xor(this, right);
            return this;
        }

        public PrimitiveColumnContainer<T> Xor(T scalar)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Xor(this, scalar);
            return this;
        }

        public PrimitiveColumnContainer<T> LeftShift(int value)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.LeftShift(this, value);
            return this;
        }

        public PrimitiveColumnContainer<T> RightShift(int value)
        {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.RightShift(this, value);
            return this;
        }

       public PrimitiveColumnContainer<T> ElementwiseEquals(PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseEquals(this, right, ret);
            return this;
       }

       public PrimitiveColumnContainer<T> ElementwiseEquals(T scalar, PrimitiveColumnContainer<bool> ret)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseEquals(this, scalar, ret);
            return this;
       }

       public PrimitiveColumnContainer<T> ElementwiseNotEquals(PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseNotEquals(this, right, ret);
            return this;
       }

       public PrimitiveColumnContainer<T> ElementwiseNotEquals(T scalar, PrimitiveColumnContainer<bool> ret)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseNotEquals(this, scalar, ret);
            return this;
       }

       public PrimitiveColumnContainer<T> ElementwiseGreaterThanOrEqual(PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseGreaterThanOrEqual(this, right, ret);
            return this;
       }

       public PrimitiveColumnContainer<T> ElementwiseGreaterThanOrEqual(T scalar, PrimitiveColumnContainer<bool> ret)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseGreaterThanOrEqual(this, scalar, ret);
            return this;
       }

       public PrimitiveColumnContainer<T> ElementwiseLessThanOrEqual(PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseLessThanOrEqual(this, right, ret);
            return this;
       }

       public PrimitiveColumnContainer<T> ElementwiseLessThanOrEqual(T scalar, PrimitiveColumnContainer<bool> ret)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseLessThanOrEqual(this, scalar, ret);
            return this;
       }

       public PrimitiveColumnContainer<T> ElementwiseGreaterThan(PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseGreaterThan(this, right, ret);
            return this;
       }

       public PrimitiveColumnContainer<T> ElementwiseGreaterThan(T scalar, PrimitiveColumnContainer<bool> ret)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseGreaterThan(this, scalar, ret);
            return this;
       }

       public PrimitiveColumnContainer<T> ElementwiseLessThan(PrimitiveColumnContainer<T> right, PrimitiveColumnContainer<bool> ret)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseLessThan(this, right, ret);
            return this;
       }

       public PrimitiveColumnContainer<T> ElementwiseLessThan(T scalar, PrimitiveColumnContainer<bool> ret)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.ElementwiseLessThan(this, scalar, ret);
            return this;
       }

       public PrimitiveColumnContainer<T> ReverseAdd(T scalar)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Add(scalar, this);
            return this;
       }
       public PrimitiveColumnContainer<T> ReverseSubtract(T scalar)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Subtract(scalar, this);
            return this;
       }
       public PrimitiveColumnContainer<T> ReverseMultiply(T scalar)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Multiply(scalar, this);
            return this;
       }
       public PrimitiveColumnContainer<T> ReverseDivide(T scalar)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Divide(scalar, this);
            return this;
       }
       public PrimitiveColumnContainer<T> ReverseModulo(T scalar)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Modulo(scalar, this);
            return this;
       }
       public PrimitiveColumnContainer<T> ReverseAnd(T scalar)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.And(scalar, this);
            return this;
       }
       public PrimitiveColumnContainer<T> ReverseOr(T scalar)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Or(scalar, this);
            return this;
       }
       public PrimitiveColumnContainer<T> ReverseXor(T scalar)
       {
            PrimitiveDataFrameColumnArithmetic<T>.Instance.Xor(scalar, this);
            return this;
       }
    }
}

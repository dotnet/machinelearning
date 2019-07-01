using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow.Eager
{
    public class Tape
    {
        public static bool IsDtypeTrainable(DataType dtype)
        {
            switch (dtype)
            {
                case DataType.DtHalf:
                case DataType.DtBfloat16:
                case DataType.DtFloat:
                case DataType.DtDouble:
                case DataType.DtComplex64:
                case DataType.DtComplex128:
                case DataType.DtResource:
                case DataType.DtVariant:
                    return true;
                default:
                    return false;
            }
        }
    }
}

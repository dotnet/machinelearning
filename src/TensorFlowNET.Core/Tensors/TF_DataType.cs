using System;
using System.Collections.Generic;
using System.Text;

namespace Tensorflow
{
    /// <summary>
    /// TF_DataType holds the type for a scalar value.  E.g., one slot in a tensor.
    /// The enum values here are identical to corresponding values in types.proto.
    /// </summary>
    public enum TF_DataType
    {
        DtInvalid = 0,
        TF_FLOAT = 1,
        TF_DOUBLE = 2,
        TF_INT32 = 3,  // Int32 tensors are always in 'host' memory.
        TF_UINT8 = 4,
        TF_INT16 = 5,
        TF_INT8 = 6,
        TF_STRING = 7,
        TF_COMPLEX64 = 8,  // Single-precision complex
        TF_COMPLEX = 8,    // Old identifier kept for API backwards compatibility
        TF_INT64 = 9,
        TF_BOOL = 10,
        TF_QINT8 = 11,     // Quantized int8
        TF_QUINT8 = 12,    // Quantized uint8
        TF_QINT32 = 13,    // Quantized int32
        TF_BFLOAT16 = 14,  // Float32 truncated to 16 bits.  Only for cast ops.
        TF_QINT16 = 15,    // Quantized int16
        TF_QUINT16 = 16,   // Quantized uint16
        TF_UINT16 = 17,
        TF_COMPLEX128 = 18,  // Double-precision complex
        TF_HALF = 19,
        TF_RESOURCE = 20,
        TF_VARIANT = 21,
        TF_UINT32 = 22,
        TF_UINT64 = 23,

        DtFloatRef = 101, // DT_FLOAT_REF
        DtDoubleRef = 102, // DT_DOUBLE_REF
        DtInt32Ref = 103, // DT_INT32_REF
    }
}

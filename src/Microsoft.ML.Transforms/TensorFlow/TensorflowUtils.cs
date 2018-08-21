// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Transforms.TensorFlow
{
    internal partial class TensorflowUtils
    {
        internal static PrimitiveType Tf2MlNetType(TFDataType type)
        {
            switch (type)
            {
                case TFDataType.Float:
                    return NumberType.R4;
                case TFDataType.Double:
                    return NumberType.R8;
                case TFDataType.Int32:
                    return NumberType.I4;
                case TFDataType.Int64:
                    return NumberType.I8;
                case TFDataType.UInt32:
                    return NumberType.U4;
                case TFDataType.UInt64:
                    return NumberType.U8;
                case TFDataType.Bool:
                    return BoolType.Instance;
                case TFDataType.String:
                    return TextType.Instance;
                default:
                    throw new NotSupportedException("Tensorflow type not supported.");
            }
        }

        public static unsafe void FetchData<T>(IntPtr data, T[] result)
        {
            var size = result.Length;

            GCHandle handle = GCHandle.Alloc(result, GCHandleType.Pinned);
            IntPtr target = handle.AddrOfPinnedObject();

            Int64 sizeInBytes = size * Marshal.SizeOf((typeof(T)));
            Buffer.MemoryCopy(data.ToPointer(), target.ToPointer(), sizeInBytes, sizeInBytes);
            handle.Free();
        }

        internal static bool IsTypeSupported(TFDataType tfoutput)
        {
            switch (tfoutput)
            {
                case TFDataType.Float:
                case TFDataType.Double:
                    return true;
                default:
                    return false;
            }
        }
    }
}

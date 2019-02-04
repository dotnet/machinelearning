// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for full license information.

using System;

namespace Microsoft.ML.Transforms.TensorFlow
{
    internal partial class TFTensor
    {
        /// <summary>
        /// Creates a tensor representing type T.
        /// The tensor will be backed with a managed-heap-allocated T.
        /// </summary>
        /// <typeparam name="T">.NET type of tensor to create</typeparam>
        /// <param name="data">value of tensor</param>
        public static TFTensor CreateScalar<T>(T data)
        {
            if (typeof(T) == typeof(System.Boolean))
            {
                return new TFTensor((System.Boolean)(object)data);
            }
            else if (typeof(T) == typeof(System.Byte))
            {
                return new TFTensor((System.Byte)(object)data);
            }
            else if (typeof(T) == typeof(System.Char))
            {
                return new TFTensor((System.Char)(object)data);
            }
            else if (typeof(T) == typeof(System.Numerics.Complex))
            {
                return new TFTensor((System.Numerics.Complex)(object)data);
            }
            else if (typeof(T) == typeof(System.Double))
            {
                return new TFTensor((System.Double)(object)data);
            }
            else if (typeof(T) == typeof(System.Single))
            {
                return new TFTensor((System.Single)(object)data);
            }
            else if (typeof(T) == typeof(System.Int32))
            {
                return new TFTensor((System.Int32)(object)data);
            }
            else if (typeof(T) == typeof(System.Int64))
            {
                return new TFTensor((System.Int64)(object)data);
            }
            else if (typeof(T) == typeof(System.SByte))
            {
                return new TFTensor((System.SByte)(object)data);
            }
            else if (typeof(T) == typeof(System.Int16))
            {
                return new TFTensor((System.Int16)(object)data);
            }
            else if (typeof(T) == typeof(System.UInt32))
            {
                return new TFTensor((System.UInt32)(object)data);
            }
            else if (typeof(T) == typeof(System.UInt64))
            {
                return new TFTensor((System.UInt64)(object)data);
            }
            else if (typeof(T) == typeof(System.UInt16))
            {
                return new TFTensor((System.UInt16)(object)data);
            }
            throw new NotSupportedException($"Unsupported type {typeof(T)}");
        }

        /// <summary>
        /// Creates a tensor representing type T[].
        /// T[] will be pinned and wrapped in a tensor.
        /// </summary>
        /// <typeparam name="T[]">.NET type of tensor to create</typeparam>
        /// <param name="data">value of tensor</param>
        /// <param name="count">The number of elements in the tensor</param>
        /// <param name="shape">shape of tensor</param>
        public static TFTensor Create<T>(T[] data, int count, TFShape shape)
        {
            if (typeof(T) == typeof(System.Boolean))
            {
                return new TFTensor(SetupTensor(TFDataType.Bool, shape, (Array)(object)data, 0, count, 4));
            }
            else if (typeof(T) == typeof(System.Byte))
            {
                return new TFTensor(SetupTensor(TFDataType.UInt8, shape, (Array)(object)data, 0, count, 1));
            }
            else if (typeof(T) == typeof(System.Char))
            {
                return new TFTensor(SetupTensor(TFDataType.UInt8, shape, (Array)(object)data, 0, count, 1));
            }
            else if (typeof(T) == typeof(System.Numerics.Complex))
            {
                return new TFTensor(SetupTensor(TFDataType.Complex128, shape, (Array)(object)data, 0, count, 16));
            }
            else if (typeof(T) == typeof(System.Double))
            {
                return new TFTensor(SetupTensor(TFDataType.Double, shape, (Array)(object)data, 0, count, 8));
            }
            else if (typeof(T) == typeof(System.Single))
            {
                return new TFTensor(SetupTensor(TFDataType.Float, shape, (Array)(object)data, 0, count, 4));
            }
            else if (typeof(T) == typeof(System.Int32))
            {
                return new TFTensor(SetupTensor(TFDataType.Int32, shape, (Array)(object)data, 0, count, 4));
            }
            else if (typeof(T) == typeof(System.Int64))
            {
                return new TFTensor(SetupTensor(TFDataType.Int64, shape, (Array)(object)data, 0, count, 8));
            }
            else if (typeof(T) == typeof(System.SByte))
            {
                return new TFTensor(SetupTensor(TFDataType.Int8, shape, (Array)(object)data, 0, count, 1));
            }
            else if (typeof(T) == typeof(System.Int16))
            {
                return new TFTensor(SetupTensor(TFDataType.Int16, shape, (Array)(object)data, 0, count, 2));
            }
            else if (typeof(T) == typeof(System.UInt32))
            {
                return new TFTensor(SetupTensor(TFDataType.UInt32, shape, (Array)(object)data, 0, count, 4));
            }
            else if (typeof(T) == typeof(System.UInt64))
            {
                return new TFTensor(SetupTensor(TFDataType.UInt64, shape, (Array)(object)data, 0, count, 8));
            }
            else if (typeof(T) == typeof(System.UInt16))
            {
                return new TFTensor(SetupTensor(TFDataType.UInt16, shape, (Array)(object)data, 0, count, 2));
            }
            // note that we will get here for jagged arrays, which is intententional since we'd need to copy them.
            throw new NotSupportedException($"Unsupported type {typeof(T)}");
        }
    }
}


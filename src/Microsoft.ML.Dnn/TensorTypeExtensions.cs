// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Internal.Utilities;
using NumSharp.Backends;
using NumSharp.Utilities;
using Tensorflow;

#if _REGEN_GLOBAL
%supported_numericals = ["Boolean","Byte","Int16","UInt16","Int32","UInt32","Int64","UInt64","Double","Single"]
%supported_numericals_lowercase = ["bool","byte","short","ushort","int","uint","long","ulong","double","float"]
%supported_numericals_TF_DataType = ["TF_BOOL","TF_UINT8","TF_INT16","TF_UINT16","TF_INT32","TF_UINT32","TF_INT64","TF_UINT64","TF_DOUBLE","TF_FLOAT"]
%supported_numericals_TF_DataType_full = ["TF_DataType.TF_BOOL","TF_DataType.TF_UINT8","TF_DataType.TF_INT16","TF_DataType.TF_UINT16","TF_DataType.TF_INT32","TF_DataType.TF_UINT32","TF_DataType.TF_INT64","TF_DataType.TF_UINT64","TF_DataType.TF_DOUBLE","TF_DataType.TF_FLOAT"]
#endif

namespace Microsoft.ML.Transforms
{
    [BestFriend]
    internal static class TensorTypeExtensions
    {
        public static void ToScalar<T>(this Tensor tensor, ref T dst) where T : unmanaged
        {
            unsafe
            {
                if (typeof(T).as_dtype() == tensor.dtype && tensor.dtype != TF_DataType.TF_STRING)
                {
                    dst = *(T*) tensor.buffer;
                    return;
                }

                //TODO When upgrading to the newest version of Tensorflow.NET, NumSharp will consequently upgrade too, just remove the second argument of the ChangeType calls.
                switch (tensor.dtype)
                {
#if _REGEN
                    %foreach supported_numericals_TF_DataType,supported_numericals,supported_numericals_lowercase%
                    case TF_DataType.#1:
                        dst = Converts.ChangeType<T>(*(#3*) tensor.buffer, NPTypeCode.#2);
                        return;
                    %
#else

                    case TF_DataType.TF_BOOL:
                        dst = Converts.ChangeType<T>(*(bool*) tensor.buffer, NPTypeCode.Boolean);
                        return;
                    case TF_DataType.TF_UINT8:
                        dst = Converts.ChangeType<T>(*(byte*) tensor.buffer, NPTypeCode.Byte);
                        return;
                    case TF_DataType.TF_INT16:
                        dst = Converts.ChangeType<T>(*(short*) tensor.buffer, NPTypeCode.Int16);
                        return;
                    case TF_DataType.TF_UINT16:
                        dst = Converts.ChangeType<T>(*(ushort*) tensor.buffer, NPTypeCode.UInt16);
                        return;
                    case TF_DataType.TF_INT32:
                        dst = Converts.ChangeType<T>(*(int*) tensor.buffer, NPTypeCode.Int32);
                        return;
                    case TF_DataType.TF_UINT32:
                        dst = Converts.ChangeType<T>(*(uint*) tensor.buffer, NPTypeCode.UInt32);
                        return;
                    case TF_DataType.TF_INT64:
                        dst = Converts.ChangeType<T>(*(long*) tensor.buffer, NPTypeCode.Int64);
                        return;
                    case TF_DataType.TF_UINT64:
                        dst = Converts.ChangeType<T>(*(ulong*) tensor.buffer, NPTypeCode.UInt64);
                        return;
                    case TF_DataType.TF_DOUBLE:
                        dst = Converts.ChangeType<T>(*(double*) tensor.buffer, NPTypeCode.Double);
                        return;
                    case TF_DataType.TF_FLOAT:
                        dst = Converts.ChangeType<T>(*(float*) tensor.buffer, NPTypeCode.Single);
                        return;
#endif

                    case TF_DataType.TF_COMPLEX64:
                    case TF_DataType.TF_COMPLEX128:
                    default:
                        throw new NotSupportedException();
                }
            }
        }

        public static void CopyTo<T>(this Tensor tensor, Span<T> destination) where T : unmanaged
        {
            unsafe
            {
                var len = checked((int) tensor.size);
                //perform regular CopyTo using Span.CopyTo.
                if (typeof(T).as_dtype() == tensor.dtype && tensor.dtype != TF_DataType.TF_STRING) //T can't be a string but tensor can.
                {
                    var src = (T*) tensor.buffer;
                    var srcSpan = new Span<T>(src, len);
                    srcSpan.CopyTo(destination);

                    return;
                }

                if (len > destination.Length)
                    throw new ArgumentException("Destinion was too short to perform CopyTo.");

                //Perform cast to type <T>.
                fixed (T* dst_ = destination)
                {
                    var dst = dst_;
                    switch (tensor.dtype)
                    {
#if _REGEN
                        %foreach supported_numericals_TF_DataType,supported_numericals,supported_numericals_lowercase%
                        case TF_DataType.#1:
                        {
                            var converter = Converts.FindConverter<#3, T>();
                            var src = (#3*) tensor.buffer;
                            Parallel.For(0, len, i => *(dst + i) = converter(unchecked(*(src + i))));
                            return;
                        }
                        %
#else

                        case TF_DataType.TF_BOOL:
                        {
                            var converter = Converts.FindConverter<bool, T>();
                            var src = (bool*) tensor.buffer;
                            Parallel.For(0, len, i => *(dst + i) = converter(unchecked(*(src + i))));
                            return;
                        }
                        case TF_DataType.TF_UINT8:
                        {
                            var converter = Converts.FindConverter<byte, T>();
                            var src = (byte*) tensor.buffer;
                            Parallel.For(0, len, i => *(dst + i) = converter(unchecked(*(src + i))));
                            return;
                        }
                        case TF_DataType.TF_INT16:
                        {
                            var converter = Converts.FindConverter<short, T>();
                            var src = (short*) tensor.buffer;
                            Parallel.For(0, len, i => *(dst + i) = converter(unchecked(*(src + i))));
                            return;
                        }
                        case TF_DataType.TF_UINT16:
                        {
                            var converter = Converts.FindConverter<ushort, T>();
                            var src = (ushort*) tensor.buffer;
                            Parallel.For(0, len, i => *(dst + i) = converter(unchecked(*(src + i))));
                            return;
                        }
                        case TF_DataType.TF_INT32:
                        {
                            var converter = Converts.FindConverter<int, T>();
                            var src = (int*) tensor.buffer;
                            Parallel.For(0, len, i => *(dst + i) = converter(unchecked(*(src + i))));
                            return;
                        }
                        case TF_DataType.TF_UINT32:
                        {
                            var converter = Converts.FindConverter<uint, T>();
                            var src = (uint*) tensor.buffer;
                            Parallel.For(0, len, i => *(dst + i) = converter(unchecked(*(src + i))));
                            return;
                        }
                        case TF_DataType.TF_INT64:
                        {
                            var converter = Converts.FindConverter<long, T>();
                            var src = (long*) tensor.buffer;
                            Parallel.For(0, len, i => *(dst + i) = converter(unchecked(*(src + i))));
                            return;
                        }
                        case TF_DataType.TF_UINT64:
                        {
                            var converter = Converts.FindConverter<ulong, T>();
                            var src = (ulong*) tensor.buffer;
                            Parallel.For(0, len, i => *(dst + i) = converter(unchecked(*(src + i))));
                            return;
                        }
                        case TF_DataType.TF_DOUBLE:
                        {
                            var converter = Converts.FindConverter<double, T>();
                            var src = (double*) tensor.buffer;
                            Parallel.For(0, len, i => *(dst + i) = converter(unchecked(*(src + i))));
                            return;
                        }
                        case TF_DataType.TF_FLOAT:
                        {
                            var converter = Converts.FindConverter<float, T>();
                            var src = (float*) tensor.buffer;
                            Parallel.For(0, len, i => *(dst + i) = converter(unchecked(*(src + i))));
                            return;
                        }
#endif
                        case TF_DataType.TF_STRING:
                        {
                            var src = tensor.StringData();
                            var culture = CultureInfo.InvariantCulture;

                            switch (typeof(T).as_dtype())
                            {
#if _REGEN
                            %foreach supported_numericals_TF_DataType,supported_numericals,supported_numericals_lowercase%
                            case TF_DataType.#1: {
                                var sdst = (#3*)Unsafe.AsPointer(ref destination.GetPinnableReference());
                                Parallel.For(0, len, i => *(sdst + i) = ((IConvertible)src[i]).To#2(culture));
                                return;
                            }
                            %
#else

                                case TF_DataType.TF_BOOL:
                                {
                                    var sdst = (bool*) Unsafe.AsPointer(ref destination.GetPinnableReference());
                                    Parallel.For(0, len, i => *(sdst + i) = ((IConvertible) src[i]).ToBoolean(culture));
                                    return;
                                }
                                case TF_DataType.TF_UINT8:
                                {
                                    var sdst = (byte*) Unsafe.AsPointer(ref destination.GetPinnableReference());
                                    Parallel.For(0, len, i => *(sdst + i) = ((IConvertible) src[i]).ToByte(culture));
                                    return;
                                }
                                case TF_DataType.TF_INT16:
                                {
                                    var sdst = (short*) Unsafe.AsPointer(ref destination.GetPinnableReference());
                                    Parallel.For(0, len, i => *(sdst + i) = ((IConvertible) src[i]).ToInt16(culture));
                                    return;
                                }
                                case TF_DataType.TF_UINT16:
                                {
                                    var sdst = (ushort*) Unsafe.AsPointer(ref destination.GetPinnableReference());
                                    Parallel.For(0, len, i => *(sdst + i) = ((IConvertible) src[i]).ToUInt16(culture));
                                    return;
                                }
                                case TF_DataType.TF_INT32:
                                {
                                    var sdst = (int*) Unsafe.AsPointer(ref destination.GetPinnableReference());
                                    Parallel.For(0, len, i => *(sdst + i) = ((IConvertible) src[i]).ToInt32(culture));
                                    return;
                                }
                                case TF_DataType.TF_UINT32:
                                {
                                    var sdst = (uint*) Unsafe.AsPointer(ref destination.GetPinnableReference());
                                    Parallel.For(0, len, i => *(sdst + i) = ((IConvertible) src[i]).ToUInt32(culture));
                                    return;
                                }
                                case TF_DataType.TF_INT64:
                                {
                                    var sdst = (long*) Unsafe.AsPointer(ref destination.GetPinnableReference());
                                    Parallel.For(0, len, i => *(sdst + i) = ((IConvertible) src[i]).ToInt64(culture));
                                    return;
                                }
                                case TF_DataType.TF_UINT64:
                                {
                                    var sdst = (ulong*) Unsafe.AsPointer(ref destination.GetPinnableReference());
                                    Parallel.For(0, len, i => *(sdst + i) = ((IConvertible) src[i]).ToUInt64(culture));
                                    return;
                                }
                                case TF_DataType.TF_DOUBLE:
                                {
                                    var sdst = (double*) Unsafe.AsPointer(ref destination.GetPinnableReference());
                                    Parallel.For(0, len, i => *(sdst + i) = ((IConvertible) src[i]).ToDouble(culture));
                                    return;
                                }
                                case TF_DataType.TF_FLOAT:
                                {
                                    var sdst = (float*) Unsafe.AsPointer(ref destination.GetPinnableReference());
                                    Parallel.For(0, len, i => *(sdst + i) = ((IConvertible) src[i]).ToSingle(culture));
                                    return;
                                }
#endif
                                default:
                                    throw new NotSupportedException();
                            }
                        }
                        case TF_DataType.TF_COMPLEX64:
                        case TF_DataType.TF_COMPLEX128:
                        default:
                            throw new NotSupportedException();
                    }
                }
            }
        }

        public static void ToArray<T>(this Tensor tensor, ref T[] array) where T : unmanaged
        {
            Utils.EnsureSize(ref array, (int) tensor.size, (int) tensor.size, false);
            var span = new Span<T>(array);

            CopyTo(tensor, span);
        }
    }
}
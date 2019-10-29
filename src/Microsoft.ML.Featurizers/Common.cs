using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace Microsoft.ML.Featurizers
{
    #region Native Function Declarations

    #endregion

    internal enum FitResult : byte
    {
        Complete = 1, Continue, ResetAndContinue
    }

    // Not all these types are currently supported. This is so the ordering will allign with the native code.
    internal enum TypeId : uint
    {
        String = 1, SByte, Short, Int, Long, Byte, UShort,
        UInt, ULong, Float16, Float32, Double, Complex64,
        Complex128, BFloat16, Bool, Timepoint, Duration,

        LastStaticValue,
        Tensor = 0x1001 | LastStaticValue + 1,
        SparseTensor = 0x1001 | LastStaticValue + 2,
        Tabular = 0x1001 | LastStaticValue + 3,
        Nullable = 0x1001 | LastStaticValue + 4,
        Vector = 0x1001 | LastStaticValue + 5,
        MapId = 0x1002 | LastStaticValue + 6
    };

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    internal unsafe struct NativeBinaryArchiveData
    {
        public byte* Data;
        public IntPtr DataSize;
    }

    #region SafeHandles

    internal class ErrorInfoSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        [DllImport("Featurizers", EntryPoint = "DestroyErrorInfo", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool DestroyErrorInfo(IntPtr error);

        public ErrorInfoSafeHandle(IntPtr handle) : base(true)
        {
            SetHandle(handle);
        }

        protected override bool ReleaseHandle()
        {
            return DestroyErrorInfo(handle);
        }
    }

    internal class ErrorInfoStringSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        [DllImport("Featurizers", EntryPoint = "DestroyErrorInfoString", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool DestroyErrorInfoString(IntPtr errorString, IntPtr errorStringSize);

        private IntPtr _length;
        public ErrorInfoStringSafeHandle(IntPtr handle, IntPtr length) : base(true)
        {
            SetHandle(handle);
            _length = length;
        }

        protected override bool ReleaseHandle()
        {
            return DestroyErrorInfoString(handle, _length);
        }
    }

    internal delegate bool DestroyTransformedDataNative(IntPtr output, IntPtr outputSize, out IntPtr errorHandle);
    internal class TransformedDataSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        private DestroyTransformedDataNative _destroySaveDataHandler;
        private IntPtr _dataSize;

        public TransformedDataSafeHandle(IntPtr handle, IntPtr dataSize, DestroyTransformedDataNative destroyCppTransformerEstimator) : base(true)
        {
            SetHandle(handle);
            _dataSize = dataSize;
            _destroySaveDataHandler = destroyCppTransformerEstimator;
        }

        protected override bool ReleaseHandle()
        {
            // Not sure what to do with error stuff here.  There shoudln't ever be one though.
            return _destroySaveDataHandler(handle, _dataSize, out IntPtr errorHandle);
        }
    }

    internal delegate bool DestroyCppTransformerEstimator(IntPtr estimator, out IntPtr errorHandle);
    internal class TransformerEstimatorSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        private DestroyCppTransformerEstimator _destroyCppTransformerEstimator;
        public TransformerEstimatorSafeHandle(IntPtr handle, DestroyCppTransformerEstimator destroyCppTransformerEstimator) : base(true)
        {
            SetHandle(handle);
            _destroyCppTransformerEstimator = destroyCppTransformerEstimator;
        }

        protected override bool ReleaseHandle()
        {
            // Not sure what to do with error stuff here. There shouldn't ever be one though.
            return _destroyCppTransformerEstimator(handle, out IntPtr errorHandle);
        }
    }

    // Destroying saved data is always the same.
    internal delegate bool DestroyTransformerSaveData(IntPtr buffer, IntPtr bufferSize, out IntPtr errorHandle);

    internal class SaveDataSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        private readonly IntPtr _dataSize;

        [DllImport("Featurizers", EntryPoint = "DestroyTransformerSaveData", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool DestroyTransformerSaveDataNative(IntPtr buffer, IntPtr bufferSize, out IntPtr error);

        public SaveDataSafeHandle(IntPtr handle, IntPtr dataSize) : base(true)
        {
            SetHandle(handle);
            _dataSize = dataSize;
        }

        protected override bool ReleaseHandle()
        {
            // Not sure what to do with error stuff here.  There shoudln't ever be one though.
            return DestroyTransformerSaveDataNative(handle, _dataSize, out _);
        }
    }

    #endregion

    internal static class CommonExtensions
    {
        [DllImport("Featurizers", EntryPoint = "GetErrorInfoString", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool GetErrorInfoString(IntPtr error, out IntPtr errorHandleString, out IntPtr errorHandleStringSize);

        internal static string GetErrorDetailsAndFreeNativeMemory(IntPtr errorHandle)
        {
            using (var error = new ErrorInfoSafeHandle(errorHandle))
            {
                GetErrorInfoString(errorHandle, out IntPtr errorHandleString, out IntPtr errorHandleStringSize);
                using (var errorString = new ErrorInfoStringSafeHandle(errorHandleString, errorHandleStringSize))
                {
                    byte[] buffer = new byte[errorHandleStringSize.ToInt32()];
                    Marshal.Copy(errorHandleString, buffer, 0, buffer.Length);

                    return Encoding.UTF8.GetString(buffer);
                }
            }
        }
        internal static TypeId GetNativeTypeIdFromType(this Type type)
        {
            if (type == typeof(byte))
                return TypeId.Byte;
            else if (type == typeof(short))
                return TypeId.Short;
            else if (type == typeof(int))
                return TypeId.Int;
            else if (type == typeof(long))
                return TypeId.Long;
            else if (type == typeof(byte))
                return TypeId.Byte;
            else if (type == typeof(ushort))
                return TypeId.UShort;
            else if (type == typeof(uint))
                return TypeId.UInt;
            else if (type == typeof(ulong))
                return TypeId.ULong;
            else if (type == typeof(float))
                return TypeId.Float32;
            else if (type == typeof(double))
                return TypeId.Double;
            else if (type == typeof(bool))
                return TypeId.Bool;
            else if (type == typeof(ReadOnlyMemory<char>))
                return TypeId.String;

            throw new InvalidOperationException($"Unsupported type {type}");
        }
    }
}

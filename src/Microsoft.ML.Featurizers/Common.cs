// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace Microsoft.ML.Featurizers
{
    // Training state of the native featurizers
    internal enum TrainingState
    {
        Pending = 1,
        Training = 2,
        Finished = 3, // Done training, but the estimator hasn't created a transformer yet
        Completed = 4 // Done training, and the estimator has created its transformer
    };

    // Fit result of the native featurizers
    internal enum FitResult : byte
    {
        Complete = 1,
        Continue = 2,
        ResetAndContinue = 3
    }

    // Not all these types are currently supported. These are taken directly from the Native code implementation.
    internal enum TypeId : uint
    {
        // Enumeration values are in the following format:
        //
        //      0xVTTTXXXX
        //        ^^^^^^^^
        //        ||  |- Id
        //        ||- Number of trailing types
        //        |- Has trailing types
        //
        String = 1,
        SByte = 2,
        Short = 3,
        Int = 4,
        Long = 5,
        Byte = 6,
        UShort = 7,
        UInt = 8,
        ULong = 9,
        Float16 = 10,
        Float32 = 11,
        Double = 12,
        Complex64 = 13,
        Complex128 = 14,
        BFloat16 = 15,
        Bool = 16,
        Timepoint = 17,
        Duration = 18,

        LastStaticValue = 19,

        // The following values have N number of trailing types
        Tensor = 0x1001 | LastStaticValue + 1,
        SparseTensor = 0x1001 | LastStaticValue + 2,
        Tabular = 0x1001 | LastStaticValue + 3,

        Nullable = 0x1001 | LastStaticValue + 4,
        Vector = 0x1001 | LastStaticValue + 5,
        MapId = 0x1002 | LastStaticValue + 6
    };

    // Is a struct mirroring the native struct.
    // I used to pass binary data between ML.NET and the native code.
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    internal unsafe struct NativeBinaryArchiveData
    {
        public byte* Data;
        public IntPtr DataSize;
    }

    #region SafeHandles

    // Safe handle that frees the memory for a native error returned to ML.NET.
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

    // Safe handle that frees the memory for errors strings return from the native code to ML.NET.
    internal class ErrorInfoStringSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        [DllImport("Featurizers", EntryPoint = "DestroyErrorInfoString", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool DestroyErrorInfoString(IntPtr errorString);

        public ErrorInfoStringSafeHandle(IntPtr handle) : base(true)
        {
            SetHandle(handle);
        }

        protected override bool ReleaseHandle()
        {
            return DestroyErrorInfoString(handle);
        }
    }

    // Safe handle that frees the memory for the transformed data.
    // Is called automatically after each call to transform.
    internal delegate bool DestroyTransformedDataNative(IntPtr output, out IntPtr errorHandle);
    internal class TransformedDataSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        private readonly DestroyTransformedDataNative _destroySaveDataHandler;

        public TransformedDataSafeHandle(IntPtr handle, DestroyTransformedDataNative destroyCppTransformerEstimator) : base(true)
        {
            SetHandle(handle);
            _destroySaveDataHandler = destroyCppTransformerEstimator;
        }

        protected override bool ReleaseHandle()
        {
            // Not sure what to do with error stuff here.  There shoudln't ever be one though.
            return _destroySaveDataHandler(handle, out IntPtr errorHandle);
        }
    }

    // Safe handle that frees the memory for a native estimator or transformer.
    // Is called automatically at the end of life for a transformer or estimator.
    internal delegate bool DestroyNativeTransformerEstimator(IntPtr estimator, out IntPtr errorHandle);
    internal class TransformerEstimatorSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
    {
        private readonly DestroyNativeTransformerEstimator _destroyNativeTransformerEstimator;
        public TransformerEstimatorSafeHandle(IntPtr handle, DestroyNativeTransformerEstimator destroyNativeTransformerEstimator) : base(true)
        {
            SetHandle(handle);
            _destroyNativeTransformerEstimator = destroyNativeTransformerEstimator;
        }

        protected override bool ReleaseHandle()
        {
            // Not sure what to do with error stuff here. There shouldn't ever be one though.
            return _destroyNativeTransformerEstimator(handle, out IntPtr errorHandle);
        }
    }

    // Safe handle that frees the memory for the internal state of a native transformer.
    // Is called automatically after we save the model.
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
            // Not sure what to do with error stuff here.  There shouldn't ever be one though.
            return DestroyTransformerSaveDataNative(handle, _dataSize, out _);
        }
    }

    #endregion

    // Static extension classes with Common methods used in multiple featurizers
    internal static class CommonExtensions
    {
        [DllImport("Featurizers", EntryPoint = "GetErrorInfoString", CallingConvention = CallingConvention.Cdecl), SuppressUnmanagedCodeSecurity]
        private static extern bool GetErrorInfoString(IntPtr error, out IntPtr errorHandleString);

        internal static string GetErrorDetailsAndFreeNativeMemory(IntPtr errorHandle)
        {
            using (var error = new ErrorInfoSafeHandle(errorHandle))
            {
                GetErrorInfoString(errorHandle, out IntPtr errorHandleString);
                using (var errorString = new ErrorInfoStringSafeHandle(errorHandleString))
                {
                    return PointerToString(errorHandleString);
                }
            }
        }

        internal static TypeId GetNativeTypeIdFromType(this Type type)
        {
            if (type == typeof(sbyte))
                return TypeId.SByte;
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

        internal static bool OsIsCentOS7()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                unsafe
                {
                    using (Process process = new Process())
                    {
                        process.StartInfo.FileName = "/bin/bash";
                        process.StartInfo.Arguments = "-c \"cat /etc/*-release\"";
                        process.StartInfo.UseShellExecute = false;
                        process.StartInfo.RedirectStandardOutput = true;
                        process.StartInfo.CreateNoWindow = true;
                        process.Start();

                        string distro = process.StandardOutput.ReadToEnd().Trim();

                        process.WaitForExit();
                        if (distro.Contains("CentOS Linux 7"))
                        {
                            return true;
                        }
                    }
                }
            return false;
        }

        internal static string PointerToString(IntPtr data)
        {
            unsafe
            {
                var length = 0;
                byte* dataPointer = (byte*)data.ToPointer();

                if (data == IntPtr.Zero)
                    return string.Empty;

                // Loop to find the size, until null is found.
                while (*dataPointer++ != 0)
                    length++;

                if (length == 0)
                    return string.Empty;

                return Encoding.UTF8.GetString((byte*)data.ToPointer(), length);
            }
        }
    }
}

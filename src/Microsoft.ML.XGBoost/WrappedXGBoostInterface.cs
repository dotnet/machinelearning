// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.Marshalling;
using System.Text;

namespace Microsoft.ML.Trainers.XGBoost
{
    /// <summary>
    /// Wrapper of the c interfaces of XGBoost
    /// Refer to https://xgboost.readthedocs.io/en/stable/tutorials/c_api_tutorial.html to get the details.
    /// </summary>

    internal static partial class WrappedXGBoostInterface
    {

        private const string DllName = "xgboost";

        [LibraryImport(DllName, EntryPoint = "XGBoostVersion")]
        public static partial void XGBoostVersion(out int major, out int minor, out int patch);

        [LibraryImport(DllName, EntryPoint = "XGBuildInfo")]
        public static unsafe partial int XGBuildInfo(byte** result);

        #region Error API 

        [LibraryImport(DllName, EntryPoint = "XGBGetLastError", StringMarshalling = StringMarshalling.Utf8)]
        public static partial string XGBGetLastError();

        #endregion

        #region DMatrix API 

        [LibraryImport(DllName, EntryPoint = "XGDMatrixCreateFromMat")]
        public static partial int XGDMatrixCreateFromMat(ReadOnlySpan<float> data, ulong nrow, ulong ncol,
                                                        float missing, out IntPtr handle);

        [LibraryImport(DllName, EntryPoint = "XGDMatrixFree")]
        public static partial int XgdMatrixFree(IntPtr handle);

        [LibraryImport(DllName, EntryPoint = "XGDMatrixNumRow")]
        public static partial int XGDMatrixNumRow(IntPtr handle, out ulong nrows);

        [LibraryImport(DllName, EntryPoint = "XGDMatrixNumCol")]
        public static partial int XGDMatrixNumCol(IntPtr handle, out ulong ncols);

        [LibraryImport(DllName, EntryPoint = "XGDMatrixGetFloatInfo", StringMarshalling = StringMarshalling.Utf8)]
#if false
        public static partial int XGDMatrixGetFloatInfo(IntPtr handle, string field,
                                                               out ulong len, [MarshalUsing(CountElementName = "len")] out Span<float> result);
#else
        public static unsafe partial int XGDMatrixGetFloatInfo(IntPtr handle, string field, out ulong length, out float* arrayPtr);
#endif

        [LibraryImport(DllName, EntryPoint = "XGDMatrixSetFloatInfo", StringMarshalling = StringMarshalling.Utf8)]
        public static partial int XGDMatrixSetFloatInfo(IntPtr handle, string field,
                                                            ReadOnlySpan<float> array, ulong len);
        #endregion


        #region API Booster

        [LibraryImport(DllName, EntryPoint = "XGBoosterCreate")]
        public static partial int XGBoosterCreate(IntPtr[] dmats,
                                                     ulong len, out IntPtr handle);

        [LibraryImport(DllName, EntryPoint = "XGBoosterFree")]
        public static partial int XGBoosterFree(IntPtr handle);

        [LibraryImport(DllName, EntryPoint = "XGBoosterSetParam", StringMarshalling = StringMarshalling.Utf8)]
        public static partial int XGBoosterSetParam(IntPtr handle, string name, string val);

        [LibraryImport(DllName, EntryPoint = "XGBoosterGetAttrNames")]
        public static unsafe partial int XGBoosterGetAttrNames(IntPtr bHandle, out ulong out_len, out byte** result);

        #endregion


        #region API train
        [LibraryImport(DllName, EntryPoint = "XGBoosterUpdateOneIter")]
        public static partial int XGBoosterUpdateOneIter(IntPtr bHandle, int iter, IntPtr dHandle);

        [DllImport(DllName)]
        public static extern int XGBoosterEvalOneIter();

        #endregion

        #region API predict
        [DllImport(DllName)]
        public static extern int XGBoosterPredict(IntPtr bHandle, IntPtr dHandle,
                                                  int optionMask, int ntreeLimit, int training,
                                                  out ulong predsLen, out IntPtr predsPtr);
        #endregion

        #region API serialization
#pragma warning disable MSML_ParameterLocalVarName
        [DllImport(DllName)]
        public static extern int XGBoosterDumpModel(IntPtr handle, string fmap, int with_stats, out int out_len, out IntPtr dumpStr);

#if false
        [DllImport(DllName)]
        public static extern int XGBoosterDumpModelEx(IntPtr handle, string fmap, int with_stats, string format, out int out_len, out IntPtr dumpStr);
#pragma warning restore MSML_ParameterLocalVarName
#endif

        [LibraryImport(DllName, EntryPoint = "XGBoosterSaveJsonConfig", StringMarshalling = StringMarshalling.Utf8)]
        public static unsafe partial int XGBoosterSaveJsonConfig(IntPtr handle, out ulong out_len, [MarshalUsing(CountElementName = "out_len")] byte** result);

        [LibraryImport(DllName, EntryPoint = "XGBoosterDumpModelEx", StringMarshalling = StringMarshalling.Utf8)]
        public static unsafe partial int XGBoosterDumpModelEx(IntPtr handle, string fmap, int with_stats, string format, out ulong out_len, out byte** result);


        #endregion

    }

    internal static class XGBoostInterfaceUtils
    {
        /// <summary>
        /// Checks if XGBoost has a pending error message. Raises an exception in that case.
        /// </summary>
        public static void Check(int res)
        {
            if (res != 0)
            {
                string mes = WrappedXGBoostInterface.XGBGetLastError();
                throw new Exception($"XGBoost Error, code is {res}, error message is '{mes}'.");
            }
        }

#if false
        public static float[] GetPredictionsArray(IntPtr predsPtr, ulong predsLen)
        {
            var length = unchecked((int)predsLen);
            var preds = new float[length];
            for (var i = 0; i < length; i++)
            {
                var floatBytes = new byte[4];
                for (var b = 0; b < 4; b++)
                {
                    floatBytes[b] = Marshal.ReadByte(predsPtr, 4 * i + b);
                }
                preds[i] = BitConverter.ToSingle(floatBytes, 0);
            }
            return preds;
        }
#endif

        /// <summary>
        /// Helper function used for generating the LightGbm argument name.
        /// When given a name, this will convert the name to lower-case with underscores.
        /// The underscore will be placed when an upper-case letter is encountered.
        /// </summary>
        public static string GetOptionName(string name)
        {
            // Otherwise convert the name to the light gbm argument
            StringBuilder strBuf = new StringBuilder();
            bool first = true;
            foreach (char c in name)
            {
                if (char.IsUpper(c))
                {
                    if (first)
                        first = false;
                    else
                        strBuf.Append('_');
                    strBuf.Append(char.ToLower(c));
                }
                else
                    strBuf.Append(c);
            }
            return strBuf.ToString();
        }

        /// <summary>
        /// Convert the pointer of c string to c# string.
        /// </summary>
        public static string GetString(IntPtr src)
        {
            return Marshal.PtrToStringAnsi(src);
        }
    }
}

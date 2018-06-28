// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace Microsoft.ML.Runtime.LightGBM
{
    /// <summary>
    /// Wrapper of the c interfaces of LightGBM.
    /// Refer to https://github.com/Microsoft/LightGBM/blob/master/include/LightGBM/c_api.h to get the details.
    /// </summary>
    internal static class WrappedLightGbmInterface
    {
        public enum CApiDType : int
        {
            Float32 = 0,
            Float64 = 1,
            Int32 = 2,
            Int64 = 3
        }

        private const string DllName = "lib_lightgbm";

        #region API Array

        [DllImport(DllName, EntryPoint = "LGBM_AllocateArray", CallingConvention = CallingConvention.StdCall)]
        public static extern int AllocateArray(
            long len,
            int type,
            ref IntPtr ret);

        [DllImport(DllName, EntryPoint = "LGBM_CopyToArray", CallingConvention = CallingConvention.StdCall)]
        public static extern int CopyToArray(
            IntPtr arr,
            int type,
            long startIdx,
            IntPtr src,
            long len);

        [DllImport(DllName, EntryPoint = "LGBM_FreeArray", CallingConvention = CallingConvention.StdCall)]
        public static extern int FreeArray(
            IntPtr ret,
            int type);

        #endregion 

        #region API ERROR

        [DllImport(DllName, EntryPoint = "LGBM_GetLastError", CallingConvention = CallingConvention.StdCall)]
        public static extern IntPtr GetLastError();

        #endregion

        #region API Dataset

        [DllImport(DllName, EntryPoint = "LGBM_DatasetCreateFromSampledColumn", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetCreateFromSampledColumn(IntPtr sampleValuePerColumn,
            IntPtr sampleIndicesPerColumn,
            int numCol,
            int[] sampleNonZeroCntPerColumn,
            int numSampleRow,
            int numTotalRow,
            [MarshalAs(UnmanagedType.LPStr)]string parameters,
            ref IntPtr ret);

        [DllImport(DllName, EntryPoint = "LGBM_DatasetCreateByReference", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetCreateByReference(IntPtr reference,
            long numRow,
            ref IntPtr ret);

        [DllImport(DllName, EntryPoint = "LGBM_DatasetPushRows", CallingConvention = CallingConvention.StdCall)]
        private static extern int DatasetPushRows(IntPtr dataset,
            float[] data,
            CApiDType dataType,
            int numRow,
            int numCol,
            int startRowIdx);

        public static int DatasetPushRows(IntPtr dataset,
            float[] data,
            int numRow,
            int numCol,
            int startRowIdx)
        {
            return DatasetPushRows(dataset, data, CApiDType.Float32, numRow, numCol, startRowIdx);
        }

        [DllImport(DllName, EntryPoint = "LGBM_DatasetPushRowsByCSR", CallingConvention = CallingConvention.StdCall)]
        private static extern int DatasetPushRowsByCsr(IntPtr dataset,
            int[] indPtr,
            CApiDType indPtrType,
            int[] indices,
            float[] data,
            CApiDType dataType,
            long nIndPtr,
            long numElem,
            long numCol,
            long startRowIdx);

        public static int DatasetPushRowsByCsr(IntPtr dataset,
            int[] indPtr,
            int[] indices,
            float[] data,
            long nIndPtr,
            long numElem,
            long numCol,
            long startRowIdx)
        {
            return DatasetPushRowsByCsr(dataset,
                indPtr, CApiDType.Int32,
                indices, data, CApiDType.Float32,
                nIndPtr, numElem, numCol, startRowIdx);
        }

        [DllImport(DllName, EntryPoint = "LGBM_DatasetFree", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetFree(IntPtr handle);

        [DllImport(DllName, EntryPoint = "LGBM_DatasetSetField", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetSetField(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPStr)]string field,
            IntPtr array,
            int len,
            CApiDType type);

        [DllImport(DllName, EntryPoint = "LGBM_DatasetGetNumData", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetGetNumData(IntPtr handle, ref int res);

        [DllImport(DllName, EntryPoint = "LGBM_DatasetGetNumFeature", CallingConvention = CallingConvention.StdCall)]
        public static extern int DatasetGetNumFeature(IntPtr handle, ref int res);

        #endregion

        #region API Booster

        [DllImport(DllName, EntryPoint = "LGBM_BoosterCreate", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterCreate(IntPtr trainset,
            [MarshalAs(UnmanagedType.LPStr)]string param,
            ref IntPtr res);

        [DllImport(DllName, EntryPoint = "LGBM_BoosterFree", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterFree(IntPtr handle);

        [DllImport(DllName, EntryPoint = "LGBM_BoosterAddValidData", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterAddValidData(IntPtr handle, IntPtr validset);

        [DllImport(DllName, EntryPoint = "LGBM_BoosterSaveModelToString", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int BoosterSaveModelToString(IntPtr handle,
            int numIteration,
            int bufferLen,
            ref int outLen,
            byte* outStr);

        #endregion

        #region API train

        [DllImport(DllName, EntryPoint = "LGBM_BoosterUpdateOneIter", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterUpdateOneIter(IntPtr handle, ref int isFinished);

        [DllImport(DllName, EntryPoint = "LGBM_BoosterGetEvalCounts", CallingConvention = CallingConvention.StdCall)]
        public static extern int BoosterGetEvalCounts(IntPtr handle, ref int outLen);

        [DllImport(DllName, EntryPoint = "LGBM_BoosterGetEval", CallingConvention = CallingConvention.StdCall)]
        public unsafe static extern int BoosterGetEval(IntPtr handle, int dataIdx,
                                 ref int outLen, double* outResult);

        #endregion

        #region API parallel

        [DllImport(DllName, EntryPoint = "LGBM_NetworkInitWithFunctions", CallingConvention = CallingConvention.StdCall)]
        public static extern int NetworkInitWithFunctions(int numMachines, int rank, ReduceScatterFunction reduceScatterFuncPtr, AllgatherFunction allgatherFuncPtr);

        [DllImport(DllName, EntryPoint = "LGBM_NetworkFree", CallingConvention = CallingConvention.StdCall)]
        public static extern int NetworkFree();

        #endregion
    }

    internal static class LightGbmInterfaceUtils
    {
        /// <summary>
        /// Checks if LightGBM has a pending error message. Raises an exception in that case.
        /// </summary>
        public static void Check(int res)
        {
            if (res != 0)
            {
                var charPtr = WrappedLightGbmInterface.GetLastError();
                string mes = Marshal.PtrToStringAnsi(charPtr);
                throw Contracts.Except("LightGBM Error, code is {0}, error message is '{1}'.", res, mes);
            }
        }

        /// <summary>
        /// Join the parameters to key=value format.
        /// </summary>
        public static string JoinParameters(Dictionary<string, string> parameters)
        {
            if (parameters == null)
                return "";
            List<string> res = new List<string>();
            foreach (var keyVal in parameters)
                res.Add(keyVal.Key + "=" + keyVal.Value);
            return string.Join(" ", res);
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

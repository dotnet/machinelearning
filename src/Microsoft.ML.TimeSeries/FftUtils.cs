// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Microsoft.ML.Runtime.TimeSeriesProcessing
{
    /// <summary>
    /// The utility functions that wrap the native Discrete Fast Fourier Transform functionality from Intel MKL.
    /// </summary>
    internal static class FftUtils
    {
        //To triger the loading of MKL library since MKL proxy native library depends on it.
        static FftUtils() => ErrorMessage(0);

        private enum ConfigParam
        {
            /* Domain for forward transform. No default value */
            ForwardDomain = 0,

            /* Dimensionality, or rank. No default value */
            Dimension = 1,

            /* Length(s) of transform. No default value */
            Lengths = 2,

            /* Floating point precision. No default value */
            Precision = 3,

            /* Scale factor for forward transform [1.0] */
            ForwardScale = 4,

            /* Scale factor for backward transform [1.0] */
            BackwardScale = 5,

            /* Exponent sign for forward transform [Negative]  */
            /* ForwardSign = 6, ## NOT IMPLEMENTED */

            /* Number of data sets to be transformed [1] */
            NumberOfTransforms = 7,

            /* Storage of finite complex-valued sequences in complex domain
               [ComplexComplex] */
            ComplexStorage = 8,

            /* Storage of finite real-valued sequences in real domain
               [RealReal] */
            RealStorage = 9,

            /* Storage of finite complex-valued sequences in conjugate-even
               domain [ComplexReal] */
            ConjugateEvenStorage = 10,

            /* Placement of result [InPlace] */
            Placement = 11,

            /* Generalized strides for input data layout [tigth, row-major for
               C] */
            InputStrides = 12,

            /* Generalized strides for output data layout [tight, row-major
               for C] */
            OutputStrides = 13,

            /* Distance between first input elements for multiple transforms
               [0] */
            InputDistance = 14,

            /* Distance between first output elements for multiple transforms
               [0] */
            OutputDistance = 15,

            /* Effort spent in initialization [Medium] */
            /* InitializationEffort = 16, ## NOT IMPLEMENTED */

            /* Use of workspace during computation [Allow] */
            /* Workspace = 17, ## NOT IMPLEMENTED */

            /* Ordering of the result [Ordered] */
            Ordering = 18,

            /* Possible transposition of result [None] */
            Transpose = 19,

            /* User-settable descriptor name [""] */
            DescriptorName = 20, /* DEPRECATED */

            /* Packing format for ComplexReal storage of finite
               conjugate-even sequences [CcsFormat] */
            PackedFormat = 21,

            /* Commit status of the descriptor - R/O parameter */
            CommitStatus = 22,

            /* Version string for this DFTI implementation - R/O parameter */
            Version = 23,

            /* Ordering of the forward transform - R/O parameter */
            /* ForwardOrdering  = 24, ## NOT IMPLEMENTED */

            /* Ordering of the backward transform - R/O parameter */
            /* BackwardOrdering = 25, ## NOT IMPLEMENTED */

            /* Number of user threads that share the descriptor [1] */
            NumberOfUserThreads = 26
        }

        private enum ConfigValue
        {
            /* CommitStatus */
            Committed = 30,
            Uncommitted = 31,

            /* ForwardDomain */
            Complex = 32,
            Real = 33,
            /* ConjugateEven = 34,   ## NOT IMPLEMENTED */

            /* Precision */
            Single = 35,
            Double = 36,

            /* ForwardSign */
            /* Negative = 37,         ## NOT IMPLEMENTED */
            /* Positive = 38,         ## NOT IMPLEMENTED */

            /* ComplexStorage and ConjugateEvenStorage */
            ComplexComplex = 39,
            ComplexReal = 40,

            /* RealStorage */
            RealComplex = 41,
            RealReal = 42,

            /* Placement */
            InPlace = 43,          /* Result overwrites input */
            NotInPlace = 44,      /* Have another place for result */

            /* InitializationEffort */
            /* Low = 45,              ## NOT IMPLEMENTED */
            /* Medium = 46,           ## NOT IMPLEMENTED */
            /* High = 47,             ## NOT IMPLEMENTED */

            /* Ordering */
            Ordered = 48,
            BackwardScrambled = 49,
            /* ForwardScrambled = 50, ## NOT IMPLEMENTED */

            /* Allow/avoid certain usages */
            Allow = 51,            /* Allow transposition or workspace */
            /* Avoid = 52,            ## NOT IMPLEMENTED */
            None = 53,

            /* PackedFormat (for storing congugate-even finite sequence
               in real array) */
            CcsFormat = 54,       /* Complex conjugate-symmetric */
            PackFormat = 55,      /* Pack format for real DFT */
            PermFormat = 56,      /* Perm format for real DFT */
            CceFormat = 57        /* Complex conjugate-even */
        }

        private const string DllName = "MklImports";
        private const string DllProxyName = "MklProxyNative";

        // See: https://software.intel.com/en-us/node/521976#8CD904AB-244B-42E4-820A-CC2376E776B8
        [DllImport(DllProxyName, EntryPoint = "MKLDftiCreateDescriptor", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        private static extern int CreateDescriptor(out IntPtr desc, ConfigValue precision, ConfigValue domain, int dimension, int length);

        // See: https://software.intel.com/en-us/node/521977
        [DllImport(DllName, EntryPoint = "DftiCommitDescriptor")]
        private static extern int CommitDescriptor(IntPtr desc);

        // See: https://software.intel.com/en-us/node/521978
        [DllImport(DllName, EntryPoint = "DftiFreeDescriptor")]
        private static extern int FreeDescriptor(ref IntPtr desc);

        // See: https://software.intel.com/en-us/node/521981
        [DllImport(DllProxyName, EntryPoint = "MKLDftiSetValue", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        private static extern int SetValue(IntPtr desc, ConfigParam configParam, ConfigValue configValue);

        // See: https://software.intel.com/en-us/node/521984
        [DllImport(DllProxyName, EntryPoint = "MKLDftiComputeForward", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        private static extern int ComputeForward(IntPtr desc, [In] double[] inputRe, [In] double[] inputIm, [Out] double[] outputRe, [Out] double[] outputIm);

        // See: https://software.intel.com/en-us/node/521985
        [DllImport(DllProxyName, EntryPoint = "MKLDftiComputeBackward", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        private static extern int ComputeBackward(IntPtr desc, [In] double[] inputRe, [In] double[] inputIm, [Out] double[] outputRe, [Out] double[] outputIm);

        // See: https://software.intel.com/en-us/node/521984
        [DllImport(DllProxyName, EntryPoint = "MKLDftiComputeForward", CallingConvention = CallingConvention.Cdecl)]
        private static extern int ComputeForward(IntPtr desc, [In] float[] inputRe, [In] float[] inputIm, [Out] float[] outputRe, [Out] float[] outputIm);

        // See: https://software.intel.com/en-us/node/521985
        [DllImport(DllProxyName, EntryPoint = "MKLDftiComputeBackward", CallingConvention = CallingConvention.Cdecl)]
        private static extern int ComputeBackward(IntPtr desc, [In] float[] inputRe, [In] float[] inputIm, [Out] float[] outputRe, [Out] float[] outputIm);

        // See: https://software.intel.com/en-us/node/521990
        [System.Security.SuppressUnmanagedCodeSecurity]
        [DllImport(DllName, EntryPoint = "DftiErrorMessage", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Auto)]
        private static extern IntPtr ErrorMessage(int status);

        private static void CheckStatus(int status)
        {
            if (status != 0)
                throw Contracts.Except(Marshal.PtrToStringAnsi(ErrorMessage(status)));
        }

        /// <summary>
        /// Computes the forward Fast Fourier Transform of the input series in single precision.
        /// </summary>
        /// <param name="inputRe">The real part of the input series</param>
        /// <param name="inputIm">The imaginary part of the input series</param>
        /// <param name="outputRe">The real part of the output series</param>
        /// <param name="outputIm">The imaginary part of the output series</param>
        /// <param name="length"></param>
        public static void ComputeForwardFft(float[] inputRe, float[] inputIm, float[] outputRe, float[] outputIm, int length)
        {
            Contracts.CheckValue(inputRe, nameof(inputRe));
            Contracts.CheckValue(inputIm, nameof(inputIm));
            Contracts.CheckValue(outputRe, nameof(outputRe));
            Contracts.CheckValue(outputIm, nameof(outputIm));
            Contracts.CheckParam(length > 0, nameof(length), "The length parameter must be greater than 0.");
            Contracts.Check(inputRe.Length >= length && inputIm.Length >= length && outputRe.Length >= length && outputIm.Length >= length,
                "The lengths of inputRe, inputIm, outputRe and outputIm need to be at least equal to the length parameter.");

            int status = 0; // DFTI_NO_ERROR
            IntPtr descriptor = default(IntPtr);

            try
            {
                status = CreateDescriptor(out descriptor, ConfigValue.Single, ConfigValue.Complex, 1, length);
                CheckStatus(status);

                status = SetValue(descriptor, ConfigParam.Placement, ConfigValue.NotInPlace);
                CheckStatus(status);

                status = SetValue(descriptor, ConfigParam.ComplexStorage, ConfigValue.RealReal);
                CheckStatus(status);

                status = CommitDescriptor(descriptor);
                CheckStatus(status);

                status = ComputeForward(descriptor, inputRe, inputIm, outputRe, outputIm);
                CheckStatus(status);
            }
            finally
            {
                if (descriptor != null)
                    FreeDescriptor(ref descriptor);
            }
        }

        /// <summary>
        /// Computes the backward (inverse) Fast Fourier Transform of the input series in single precision.
        /// </summary>
        /// <param name="inputRe">The real part of the input series</param>
        /// <param name="inputIm">The imaginary part of the input series</param>
        /// <param name="outputRe">The real part of the output series</param>
        /// <param name="outputIm">The imaginary part of the output series</param>
        /// <param name="length"></param>
        public static void ComputeBackwardFft(float[] inputRe, float[] inputIm, float[] outputRe, float[] outputIm, int length)
        {
            Contracts.CheckValue(inputRe, nameof(inputRe));
            Contracts.CheckValue(inputIm, nameof(inputIm));
            Contracts.CheckValue(outputRe, nameof(outputRe));
            Contracts.CheckValue(outputIm, nameof(outputIm));
            Contracts.CheckParam(length > 0, nameof(length), "The length parameter must be greater than 0.");
            Contracts.Check(inputRe.Length >= length && inputIm.Length >= length && outputRe.Length >= length && outputIm.Length >= length,
                "The lengths of inputRe, inputIm, outputRe and outputIm need to be at least equal to the length parameter.");

            int status = 0; // DFTI_NO_ERROR
            IntPtr descriptor = default(IntPtr);
            float scale = 1f / length;

            try
            {
                status = CreateDescriptor(out descriptor, ConfigValue.Single, ConfigValue.Complex, 1, length);
                CheckStatus(status);

                status = SetValue(descriptor, ConfigParam.Placement, ConfigValue.NotInPlace);
                CheckStatus(status);

                status = SetValue(descriptor, ConfigParam.ComplexStorage, ConfigValue.RealReal);
                CheckStatus(status);

                status = CommitDescriptor(descriptor);
                CheckStatus(status);

                status = ComputeBackward(descriptor, inputRe, inputIm, outputRe, outputIm);
                CheckStatus(status);
            }
            finally
            {
                if (descriptor != null)
                    FreeDescriptor(ref descriptor);
            }

            // REVIEW: for some reason the native backward scaling for DFTI in MKL does not work.
            // Therefore here, we manually re-scale the output.
            // Ideally, the command
            // status = SetValue(descriptor, ConfigParam.BackwardScale, __arglist(scale));
            // should do the backward rescaling but for some reason it does not work and needs further investigation.
            for (int i = 0; i < length; ++i)
            {
                outputRe[i] *= scale;
                outputIm[i] *= scale;
            }
        }

        /// <summary>
        /// Computes the forward Fast Fourier Transform of the input series in double precision.
        /// </summary>
        /// <param name="inputRe">The real part of the input series</param>
        /// <param name="inputIm">The imaginary part of the input series</param>
        /// <param name="outputRe">The real part of the output series</param>
        /// <param name="outputIm">The imaginary part of the output series</param>
        /// <param name="length"></param>
        public static void ComputeForwardFft(double[] inputRe, double[] inputIm, double[] outputRe, double[] outputIm, int length)
        {
            Contracts.CheckValue(inputRe, nameof(inputRe));
            Contracts.CheckValue(inputIm, nameof(inputIm));
            Contracts.CheckValue(outputRe, nameof(outputRe));
            Contracts.CheckValue(outputIm, nameof(outputIm));
            Contracts.CheckParam(length > 0, nameof(length), "The length parameter must be greater than 0.");
            Contracts.Check(inputRe.Length >= length && inputIm.Length >= length && outputRe.Length >= length && outputIm.Length >= length,
                "The lengths of inputRe, inputIm, outputRe and outputIm need to be at least equal to the length parameter.");

            int status = 0; // DFTI_NO_ERROR
            IntPtr descriptor = default(IntPtr);

            try
            {
                status = CreateDescriptor(out descriptor, ConfigValue.Double, ConfigValue.Complex, 1, length);
                CheckStatus(status);

                status = SetValue(descriptor, ConfigParam.Placement, ConfigValue.NotInPlace);
                CheckStatus(status);

                status = SetValue(descriptor, ConfigParam.ComplexStorage, ConfigValue.RealReal);
                CheckStatus(status);

                status = CommitDescriptor(descriptor);
                CheckStatus(status);

                status = ComputeForward(descriptor, inputRe, inputIm, outputRe, outputIm);
                CheckStatus(status);
            }
            finally
            {
                if (descriptor != null)
                    FreeDescriptor(ref descriptor);
            }
        }

        /// <summary>
        /// Computes the backward (inverse) Fast Fourier Transform of the input series in double precision.
        /// </summary>
        /// <param name="inputRe">The real part of the input series</param>
        /// <param name="inputIm">The imaginary part of the input series</param>
        /// <param name="outputRe">The real part of the output series</param>
        /// <param name="outputIm">The imaginary part of the output series</param>
        /// <param name="length"></param>
        public static void ComputeBackwardFft(double[] inputRe, double[] inputIm, double[] outputRe, double[] outputIm, int length)
        {
            Contracts.CheckValue(inputRe, nameof(inputRe));
            Contracts.CheckValue(inputIm, nameof(inputIm));
            Contracts.CheckValue(outputRe, nameof(outputRe));
            Contracts.CheckValue(outputIm, nameof(outputIm));
            Contracts.CheckParam(length > 0, nameof(length), "The length parameter must be greater than 0.");
            Contracts.Check(inputRe.Length >= length && inputIm.Length >= length && outputRe.Length >= length && outputIm.Length >= length,
                "The lengths of inputRe, inputIm, outputRe and outputIm need to be at least equal to the length parameter.");

            int status = 0; // DFTI_NO_ERROR
            IntPtr descriptor = default(IntPtr);
            double scale = 1.0 / length;

            try
            {
                status = CreateDescriptor(out descriptor, ConfigValue.Double, ConfigValue.Complex, 1, length);
                CheckStatus(status);

                status = SetValue(descriptor, ConfigParam.Placement, ConfigValue.NotInPlace);
                CheckStatus(status);

                status = SetValue(descriptor, ConfigParam.ComplexStorage, ConfigValue.RealReal);
                CheckStatus(status);

                status = CommitDescriptor(descriptor);
                CheckStatus(status);

                status = ComputeBackward(descriptor, inputRe, inputIm, outputRe, outputIm);
                CheckStatus(status);
            }
            finally
            {
                if (descriptor != null)
                    FreeDescriptor(ref descriptor);
            }

            // REVIEW: for some reason the native backward scaling for DFTI in MKL does not work.
            // Therefore here, we manually re-scale the output.
            // Ideally, the command
            // status = SetValue(descriptor, ConfigParam.BackwardScale, __arglist(scale));
            // should do the backward rescaling but for some reason it does not work and needs further investigation.
            for (int i = 0; i < length; ++i)
            {
                outputRe[i] *= scale;
                outputIm[i] *= scale;
            }
        }
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Microsoft.ML.Runtime.EntryPoints;

namespace Microsoft.ML.Runtime.LightGBM
{
    /// <summary>
    /// Signature of LightGBM IAllreduce
    /// </summary>
    public delegate void SignatureParallelTrainer();

    /// <summary>
    /// Reduce function define in LightGBM Cpp side
    /// </summary>
    public unsafe delegate void ReduceFunction(byte* src, byte* output, int typeSize, int arraySize);

    /// <summary>
    /// Definition of ReduceScatter funtion
    /// </summary>
    public delegate void ReduceScatterFunction([MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)]byte[] input, int inputSize, int typeSize,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 5)]int[] blockStart, [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 5)]int[] blockLen, int numBlock,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 7)]byte[] output, int outputSize,
        IntPtr reducer);

    /// <summary>
    /// Definition of Allgather funtion
    /// </summary>
    public delegate void AllgatherFunction([MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)]byte[] input, int inputSize,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 4)]int[] blockStart, [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 4)]int[] blockLen, int numBlock,
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 6)]byte[] output, int outputSize);

    public interface IParallel
    {
        /// <summary>
        /// Type of parallel
        /// </summary>
        string ParallelType();

        /// <summary>
        /// Number of machines
        /// </summary>
        int NumMachines();

        /// <summary>
        /// Rank of local machine
        /// </summary>
        int Rank();

        /// <summary>
        /// ReduceScatter Function
        /// </summary>
        ReduceScatterFunction GetReduceScatterFunction();

        /// <summary>
        /// Allgather Function
        /// </summary>
        AllgatherFunction GetAllgatherFunction();

        /// <summary>
        /// Additional parameteres
        /// </summary>
        Dictionary<string, string> AdditionalParams();
    }

    [TlcModule.ComponentKind("ParallelLightGBM")]
    public interface ISupportParallel : IComponentFactory<IParallel>
    {
    }
}

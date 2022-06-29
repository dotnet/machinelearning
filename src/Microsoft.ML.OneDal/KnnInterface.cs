// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;

using System.IO;
using System.Linq;

using System.Runtime.InteropServices;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

using Microsoft.Win32.SafeHandles;

namespace Microsoft.ML.OneDal
{
#if false
    internal static class KnnInterface
    {
        public sealed class SafeKnnAlgorithmHandle : SafeHandleZeroOrMinusOneIsInvalid
        {
            private SafeKnnAlgorithmHandle()
              : base(true)
            {
                /* empty */
            }

            protected override bool ReleaseHandle()
            {
                DestroyHandle(handle);
                return true;
            }
        }

#if _WINDOWS
    const string libDirPath = @"C:\Users\rgesteve\Documents\projects\onedal_simple\build";
    const string libPath = libDirPath + @"\Debug\OneDALNative_lib.dll";
#else
        //const string libDirPath = @"/data/Documents/Snippets/onedal/first/build";
        private const string LibDirPath = @"/home/rgesteve/projects/onedal_simple/build";
        private const string LibPath = LibDirPath + "/libOneDALNative_lib.so";
#endif

        [DllImport(LibPath)]
        public static extern unsafe SafeKnnAlgorithmHandle CreateEngine(int numClasses);

        [DllImport(LibPath)]
        private static extern unsafe void DestroyHandle(IntPtr algorithm);

        [DllImport(LibPath)]
        public static extern unsafe int CreateKNNTable(SafeKnnAlgorithmHandle engine, void* block, int numCols, int numRows);

        [DllImport(LibPath)]
        public static extern unsafe void Train(SafeKnnAlgorithmHandle engine, void* trainingData, void* labelData, int numCols, int numRows);

        [DllImport(LibPath)]
        public static extern unsafe float SanityCheckBlock(SafeKnnAlgorithmHandle engine, void* block, int blockSize, void* outputArray);

    }

    public sealed class KnnAlgorithm : IDisposable
    {
        private readonly KnnInterface.SafeKnnAlgorithmHandle _engine;
        private bool _isDisposed;

        public KnnAlgorithm(int numClasses)
        {
            _engine = KnnInterface.CreateEngine(numClasses);
        }

#if false
    public float SanityCheckBlock(float[] block, float[] outData)
    {
        float ret = default(float);
        unsafe
        {
            fixed (void* dataPtr = &block[0], outputPtr = &outData[0])
            {
                ret = KNNInterface.SanityCheckBlock(_engine, dataPtr, block.Length, outputPtr);  // not sure if I should return from inside a fixed block
            }
        }
        return ret;
    }
#endif

        public int CreateTable(float[] block, int numCols, int numRows)
        {
            int ret = default(int);
            unsafe
            {
                fixed (void* dataPtr = &block[0])
                {
                    ret = KnnInterface.CreateKNNTable(_engine, dataPtr, numCols, numRows);
                }
            }
            return ret;
        }

        public void Train(float[] trainData, float[] labelData, int numTrainingCols, int numRows)
        {
            unsafe
            {
#pragma warning disable MSML_SingleVariableDeclaration // Have only a single variable present per declaration
                fixed (void* trainDataPtr = &trainData[0],
                             labelDataPtr = &labelData[0]) {
                    KnnInterface.Train(_engine, trainDataPtr, labelDataPtr, numTrainingCols, numRows);
                }
#pragma warning restore MSML_SingleVariableDeclaration // Have only a single variable present per declaration
            }
        }

        public void Dispose()
        {
            if (_isDisposed) return;
            _isDisposed = true;
            _engine.Dispose();
        }
    }
#endif
}

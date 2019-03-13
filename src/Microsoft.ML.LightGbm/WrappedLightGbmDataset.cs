// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.LightGbm
{
    /// <summary>
    /// Wrapper of Dataset object of LightGBM.
    /// </summary>
    internal sealed class Dataset : IDisposable
    {
        private IntPtr _handle;
        private int _lastPushedRowID;
        public IntPtr Handle => _handle;

        /// <summary>
        /// Create a <see cref="Dataset"/> for storing training and prediciton data under LightGBM framework. The main goal of this function
        /// is not marshaling ML.NET data set into LightGBM format but just creates a (unmanaged) container where examples can be pushed into by calling
        /// <see cref="PushRows(float[], int, int, int)"/>. It also pre-allocates memory so the actual size (number of examples and number of features)
        /// of the data set is required. A sub-sampled version of the original data set is passed in to compute some statictics needed by the training
        /// procedure. Note that we use "original" to indicate a property from the unsampled data set.
        /// </summary>
        /// <param name="sampleValuePerColumn">A 2-D array which encodes the sub-sampled data matrix. sampleValuePerColumn[i] stores
        /// all the non-zero values of the i-th feature. sampleValuePerColumn[i][j] is the j-th non-zero value of i-th feature encountered when scanning
        /// the values row-by-row (i.e., example-by-example) in the matrix and column-by-column (i.e., feature-by-feature) within one row. It is similar
        /// to CSC format for storing sparse matrix.</param>
        /// <param name="sampleIndicesPerColumn">A 2-D array which encodes sub-sampled example indexes of non-zero features stored in sampleValuePerColumn.
        /// The sampleIndicesPerColumn[i][j]-th example has a non-zero i-th feature whose value is sampleValuePerColumn[i][j].</param>
        /// <param name="numCol">Total number of features in the original data.</param>
        /// <param name="sampleNonZeroCntPerColumn">sampleNonZeroCntPerColumn[i] is the size of sampleValuePerColumn[i].</param>
        /// <param name="numSampleRow">The number of sampled examples in the sub-sampled data matrix.</param>
        /// <param name="numTotalRow">The number of original examples added using <see cref="PushRows(float[], int, int, int)"/>.</param>
        /// <param name="param">LightGBM parameter used in https://github.com/Microsoft/LightGBM/blob/c920e6345bcb41fc1ec6ac338f5437034b9f0d38/src/c_api.cpp#L421. </param>
        /// <param name="labels">Labels of the original data. labels[i] is the label of the i-th original example.</param>
        /// <param name="weights">Example weights of the original data. weights[i] is the weight of the i-th original example.</param>
        /// <param name="groups">Group identifiers of the original data. groups[i] is the group ID of the i-th original example.</param>
        public unsafe Dataset(double[][] sampleValuePerColumn,
            int[][] sampleIndicesPerColumn,
            int numCol,
            int[] sampleNonZeroCntPerColumn,
            int numSampleRow,
            int numTotalRow,
            string param, float[] labels, float[] weights = null, int[] groups = null)
        {
            _handle = IntPtr.Zero;

            // Use GCHandle to pin the memory, avoid the memory relocation.
            GCHandle[] gcValues = new GCHandle[numCol];
            GCHandle[] gcIndices = new GCHandle[numCol];
            try
            {
                double*[] ptrArrayValues = new double*[numCol];
                int*[] ptrArrayIndices = new int*[numCol];
                for (int i = 0; i < numCol; i++)
                {
                    gcValues[i] = GCHandle.Alloc(sampleValuePerColumn[i], GCHandleType.Pinned);
                    ptrArrayValues[i] = (double*)gcValues[i].AddrOfPinnedObject().ToPointer();
                    gcIndices[i] = GCHandle.Alloc(sampleIndicesPerColumn[i], GCHandleType.Pinned);
                    ptrArrayIndices[i] = (int*)gcIndices[i].AddrOfPinnedObject().ToPointer();
                };
                fixed (double** ptrValues = ptrArrayValues)
                fixed (int** ptrIndices = ptrArrayIndices)
                {
                    // Create container. Examples will pushed in later.
                    LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.DatasetCreateFromSampledColumn(
                        (IntPtr)ptrValues, (IntPtr)ptrIndices, numCol, sampleNonZeroCntPerColumn, numSampleRow, numTotalRow,
                        param, ref _handle));
                }
            }
            finally
            {
                for (int i = 0; i < numCol; i++)
                {
                    if (gcValues[i].IsAllocated)
                        gcValues[i].Free();
                    if (gcIndices[i].IsAllocated)
                        gcIndices[i].Free();
                };
            }
            // Before adding examples (i.e., feature vectors of the original data set), the original labels, weights, and groups are added.
            SetLabel(labels);
            SetWeights(weights);
            SetGroup(groups);

            Contracts.Assert(GetNumCols() == numCol);
            Contracts.Assert(GetNumRows() == numTotalRow);
        }

        public Dataset(Dataset reference, int numTotalRow, float[] labels, float[] weights = null, int[] groups = null)
        {
            IntPtr refHandle = IntPtr.Zero;
            if (reference != null)
                refHandle = reference.Handle;

            LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.DatasetCreateByReference(refHandle, numTotalRow, ref _handle));

            SetLabel(labels);
            SetWeights(weights);
            SetGroup(groups);
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
                LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.DatasetFree(_handle));
            _handle = IntPtr.Zero;
        }

        /// <summary>
        /// Append examples to LightGBM dataset.
        /// </summary>
        /// <param name="data">Dense (# of rows)-by-(# of columns) matrix flattened in a row-major format. One row per example.
        /// The value at the i-th row and j-th column is stored in data[j + i * (# of columns)].</param>
        /// <param name="numRow"># of rows of the data matrix.</param>
        /// <param name="numCol"># of columns of the data matrix.</param>
        /// <param name="startRowIdx">The actual row index of the first row pushed in. If it's 36, the first row in data would be the 37th row in <see cref="Dataset"/>.</param>
        public void PushRows(float[] data, int numRow, int numCol, int startRowIdx)
        {
            Contracts.Assert(startRowIdx == _lastPushedRowID);
            Contracts.Assert(numCol == GetNumCols());
            Contracts.Assert(numRow > 0);
            Contracts.Assert(startRowIdx <= GetNumRows() - numRow);
            LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.DatasetPushRows(_handle, data, numRow, numCol, startRowIdx));
            _lastPushedRowID = startRowIdx + numRow;
        }

        public void PushRows(int[] indPtr, int[] indices, float[] data, int nIndptr,
            long numElem, int numCol, int startRowIdx)
        {
            Contracts.Assert(startRowIdx == _lastPushedRowID);
            Contracts.Assert(numCol == GetNumCols());
            Contracts.Assert(startRowIdx < GetNumRows());
            LightGbmInterfaceUtils.Check(
                WrappedLightGbmInterface.DatasetPushRowsByCsr(
                    _handle, indPtr, indices, data, nIndptr, numElem, numCol, startRowIdx));
            _lastPushedRowID = startRowIdx + nIndptr - 1;
        }

        public int GetNumRows()
        {
            int res = 0;
            LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.DatasetGetNumData(_handle, ref res));
            return res;
        }

        public int GetNumCols()
        {
            int res = 0;
            LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.DatasetGetNumFeature(_handle, ref res));
            return res;
        }

        public unsafe void SetLabel(float[] labels)
        {
            Contracts.AssertValue(labels);
            Contracts.Assert(labels.Length == GetNumRows());
            fixed (float* ptr = labels)
                LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.DatasetSetField(_handle, "label", (IntPtr)ptr, labels.Length,
                    WrappedLightGbmInterface.CApiDType.Float32));
        }

        public unsafe void SetWeights(float[] weights)
        {
            if (weights != null)
            {
                Contracts.Assert(weights.Length == GetNumRows());
                // Skip SetWeights if all weights are same.
                bool allSame = true;
                for (int i = 1; i < weights.Length; ++i)
                {
                    if (weights[i] != weights[0])
                    {
                        allSame = false;
                        break;
                    }
                }
                if (!allSame)
                {
                    fixed (float* ptr = weights)
                        LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.DatasetSetField(_handle, "weight", (IntPtr)ptr, weights.Length,
                            WrappedLightGbmInterface.CApiDType.Float32));
                }
            }
        }

        public unsafe void SetGroup(int[] groups)
        {
            if (groups != null)
            {
                fixed (int* ptr = groups)
                    LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.DatasetSetField(_handle, "group", (IntPtr)ptr, groups.Length,
                        WrappedLightGbmInterface.CApiDType.Int32));
            }
        }

        // Not used now. Can use for the continued train.
        public unsafe void SetInitScore(double[] initScores)
        {
            if (initScores != null)
            {
                Contracts.Assert(initScores.Length % GetNumRows() == 0);
                fixed (double* ptr = initScores)
                    LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.DatasetSetField(_handle, "init_score", (IntPtr)ptr, initScores.Length,
                        WrappedLightGbmInterface.CApiDType.Float64));
            }
        }
    }
}

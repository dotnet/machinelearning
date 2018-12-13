// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace Microsoft.ML.Runtime.LightGBM
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
        /// Create a <see cref="Dataset"/> for training and prediciton under LightGBM framework.
        /// </summary>
        /// <param name="sampleValuePerColumn">A 2-D array which encodes the whole data matrix. sampleValuePerColumn[i] stores
        /// the non-zero values of the i-th feature. It is similar to CSC format for storing sparse matrix.</param>
        /// <param name="sampleIndicesPerColumn">A 2-D array which encodes example indexes of non-zero features stored in sampleValuePerColumn.
        /// The sampleIndicesPerColumn[i][j]-th example has a non-zero i-th feature whose value is sampleValuePerColumn[i][j].</param>
        /// <param name="numCol">Total number of features in the training data. Its the first dimension of sampleValuePerColumn;
        /// that is, the last column in the data is sampleValuePerColumn[numCol-1].</param>
        /// <param name="sampleNonZeroCntPerColumn">sampleNonZeroCntPerColumn[i] is the size of sampleValuePerColumn[i].</param>
        /// <param name="numSampleRow">The number of sampled examples when creating the training matrix using sampling without replacement.
        /// It's identical to numTotalRow if sampling.</param>
        /// <param name="numTotalRow">The number of total training examples.</param>
        /// <param name="param">LightGBM parameter used at https://github.com/Microsoft/LightGBM/blob/c920e6345bcb41fc1ec6ac338f5437034b9f0d38/src/c_api.cpp#L421. </param>
        /// <param name="labels">Labels of the original data set. It contains numTotalRow values.</param>
        /// <param name="weights">Example weights of the original data set. It contains numTotalRow values.</param>
        /// <param name="groups">Group identifiers of the original data set. It contains numTotalRow values.</param>
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
                for(int i = 0; i < numCol; i++)
                {
                    gcValues[i] = GCHandle.Alloc(sampleValuePerColumn[i], GCHandleType.Pinned);
                    ptrArrayValues[i] = (double*)gcValues[i].AddrOfPinnedObject().ToPointer();
                    gcIndices[i] = GCHandle.Alloc(sampleIndicesPerColumn[i], GCHandleType.Pinned);
                    ptrArrayIndices[i] = (int*)gcIndices[i].AddrOfPinnedObject().ToPointer();
                };
                fixed (double** ptrValues = ptrArrayValues)
                fixed (int** ptrIndices = ptrArrayIndices)
                {
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

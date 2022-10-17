// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.Trainers.XGBoost
{
    /// <summary>
    /// Wrapper of DMatrix object of XGBoost
    /// </summary>
    internal sealed class DMatrix : IDisposable
    {
        private bool disposed = false;
        private IntPtr _handle;
        public IntPtr Handle => _handle;
        private const float Missing = 0f;

        /// <summary>
        /// Create a <see cref="DMatrix"/> for storing training and prediction data under XGBoost framework.
        /// </summary>
        public unsafe DMatrix(float[] data, uint nrows, uint ncols, float[]? labels = null)
        {
            int errp = WrappedXGBoostInterface.XGDMatrixCreateFromMat(data, nrows, ncols, Missing, out _handle);
            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }

            if (labels != null)
            {
                SetLabel(labels);
            }

        }

        public ulong GetNumRows()
        {
            ulong numRows;
            int errp = WrappedXGBoostInterface.XGDMatrixNumRow(_handle, out numRows);
            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }
            return numRows;
        }

        public ulong GetNumCols()
        {
            ulong numCols;
            int errp = WrappedXGBoostInterface.XGDMatrixNumCol(_handle, out numCols);
            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }
            return numCols;
        }

        public unsafe void SetLabel(float[] labels)
        {
#if false
            Contracts.AssertValue(labels);
            Contracts.Assert(labels.Length == GetNumRows());
#endif
            fixed (float* ptr = labels)
            {
                int errp = WrappedXGBoostInterface.XGDMatrixSetFloatInfo(_handle, "label", (IntPtr)ptr, (ulong)labels.Length);
                if (errp == -1)
                {
                    string reason = WrappedXGBoostInterface.XGBGetLastError();
                    throw new XGBoostDLLException(reason);
                }
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        void Dispose(bool disposing)
        {
            if (disposed)
            {
                return;
            }

            int errp = WrappedXGBoostInterface.XGDMatrixFree(_handle);
            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }
            disposed = true;

        }
    }
}


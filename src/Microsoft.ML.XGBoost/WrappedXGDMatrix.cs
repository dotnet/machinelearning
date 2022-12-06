// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Runtime.InteropServices;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.XGBoost
{
    /// <summary>
    /// Wrapper of DMatrix object of XGBoost
    /// </summary>
    internal sealed class DMatrix : IDisposable
    {
#pragma warning disable MSML_PrivateFieldName
        private bool disposed = false;
#pragma warning restore MSML_PrivateFieldName
#pragma warning disable IDE0044
        private IntPtr _handle;
#pragma warning restore IDE0044
        public IntPtr Handle => _handle;
        private const float Missing = 0f;

        /// <summary>
        /// Create a <see cref="DMatrix"/> for storing training and prediction data under XGBoost framework.
        /// </summary>
#nullable enable
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
#nullable disable

        public int GetNumRows()
        {
            ulong numRows;
            int errp = WrappedXGBoostInterface.XGDMatrixNumRow(_handle, out numRows);
            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }
            return (int)numRows;
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

        public void SetLabel(float[] labels)
        {
            Contracts.AssertValue(labels);
            Contracts.Assert(labels.Length == GetNumRows());

            int errp = WrappedXGBoostInterface.XGDMatrixSetFloatInfo(_handle, "label", labels, (ulong)labels.Length);
            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }
        }

        public Span<float> GetLabels()
        {
            int errp = WrappedXGBoostInterface.XGDMatrixGetFloatInfo(_handle, "label", out ulong labellen, out Span<float> result);
            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }
            return result;
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

            int errp = WrappedXGBoostInterface.XgdMatrixFree(_handle);
            if (errp == -1)
            {
                string reason = WrappedXGBoostInterface.XGBGetLastError();
                throw new XGBoostDLLException(reason);
            }
            disposed = true;

        }
    }
}


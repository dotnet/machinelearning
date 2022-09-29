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

        private WrappedXGBoostInterface.SafeDMatrixHandle _handle;
        public WrappedXGBoostInterface.SafeDMatrixHandle Handle => _handle;
	private const float Missing = 0f;  // Value to use for Missing values in matrices

        /// <summary>
        /// Create a <see cref="DMatrix"/> for storing training and prediction data under XGBoost framework.
        public unsafe DMatrix(float[] data, uint nrows, uint ncols, float[] labels = null)
	{
	  _handle = null;

	  int errp = WrappedXGBoostInterface.XGDMatrixCreateFromMat(data, nrows, ncols, Missing, out _handle);
	  if (errp == -1)
	  {
	      Contracts.Except(WrappedXGBoostInterface.XGBGetLastError());
	  }

	}

        public void Dispose()
        {
            _handle?.Dispose();
            _handle = null;
        }

	public ulong GetNumRows()
	{
	  ulong numRows;
	  int errp = WrappedXGBoostInterface.XGDMatrixNumRow(_handle, out numRows);
	  if (errp == -1) {
	      Contracts.Except(WrappedXGBoostInterface.XGBGetLastError());
	  }
	  return numRows;
	}

	public ulong GetNumCols()
	{
	  ulong numCols;
	  int errp = WrappedXGBoostInterface.XGDMatrixNumCol(_handle, out numCols);
	  if (errp == -1) {
	      Contracts.Except(WrappedXGBoostInterface.XGBGetLastError());
	  }
	  return numCols;
	}

        public unsafe void SetLabel(float[] labels)
        {
            Contracts.AssertValue(labels);
#if false
            Contracts.Assert(labels.Length == GetNumRows());
            fixed (float* ptr = labels)
                LightGbmInterfaceUtils.Check(WrappedLightGbmInterface.DatasetSetField(_handle, "label", (IntPtr)ptr, labels.Length,
                    WrappedLightGbmInterface.CApiDType.Float32));
#endif
        }
    }
}

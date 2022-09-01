//#include <cstdio>
//#include <cstdlib>
#include <xgboost/c_api.h>
#include "../Stdafx.h"

EXPORT_API(void) xgboostClfToDMatrix(void* featuresPtr, void* labels, void* results, long long nRows, long long nCols)
{
  DMatrixHandle dmatrix;
  XGDMatrixCreateFromMat((float*)featuresPtr, nRows, nCols, 0, &dmatrix);

  return;
}

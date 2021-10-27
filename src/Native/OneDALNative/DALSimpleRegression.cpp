// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#include <stdlib.h>

#include "DALSimpleRegression.h"

EXPORT_API(void) linearRegressionSingle(void * features, void * label, void * betas, int nRows, int nColumns)
{
  int nRows_local = nRows;
  int nRColumns_local = 0;
  if (nRows_local > 10) {
    int nRColumns_local = nColumns;
  }
  return;
}


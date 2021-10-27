// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#include <stdlib.h>

#include "daal.h"

#include "DALSimpleRegression.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

#if 0
EXPORT_API(void) linearRegressionSingle(void * features, void * label, void * betas, int nRows, int nColumns)
{
  int nRows_local = nRows;
  int nRColumns_local = 0;
  if (nRows_local > 10) {
    int nRColumns_local = nColumns;
  }
  return;
}
#endif

template <typename FPType>
void linearRegression(FPType * features, FPType * label, FPType * betas, int nRows, int nColumns)
{
    NumericTablePtr featuresTable(new HomogenNumericTable<FPType>(features, nColumns, nRows));
    NumericTablePtr labelsTable(new HomogenNumericTable<FPType>(label, 1, nRows));

    // Training
    linear_regression::training::Batch<FPType> trainingAlgorithm;
    trainingAlgorithm.input.set(linear_regression::training::data, featuresTable);
    trainingAlgorithm.input.set(linear_regression::training::dependentVariables, labelsTable);
    trainingAlgorithm.compute();
    linear_regression::training::ResultPtr trainingResult = trainingAlgorithm.getResult();

    // Betas copying
    NumericTablePtr betasTable = trainingResult->get(linear_regression::training::model)->getBeta();
    BlockDescriptor<FPType> betasBlock;
    betasTable->getBlockOfRows(0, 1, readWrite, betasBlock);
    FPType * betasForCopy = betasBlock.getBlockPtr();
    for (int i = 0; i < nColumns + 1; ++i)
    {
        betas[i] = betasForCopy[i];
    }
}


template <typename FPType>
int ridgeRegressionOnlineComputeTemplate(FPType * featuresPtr, FPType * labelsPtr, int nRows, int nColumns, float l2Reg, byte * partialResultPtr, int partialResultSize)
{
    // Create input data tables
    NumericTablePtr featuresTable(new HomogenNumericTable<FPType>(featuresPtr, nColumns, nRows));
    NumericTablePtr labelsTable(new HomogenNumericTable<FPType>(labelsPtr, 1, nRows));
    FPType l2 = l2Reg;
    NumericTablePtr l2RegTable(new HomogenNumericTable<FPType>(&l2, 1, 1));

    // Set up and execute training
    ridge_regression::training::Online<FPType> trainingAlgorithm;
    trainingAlgorithm.parameter.ridgeParameters = l2RegTable;

    ridge_regression::training::PartialResultPtr pRes(new ridge_regression::training::PartialResult);
    if (partialResultSize != 0)
    {
        OutputDataArchive dataArch(partialResultPtr, partialResultSize);
        pRes->deserialize(dataArch);
        trainingAlgorithm.setPartialResult(pRes);
    }

    trainingAlgorithm.input.set(ridge_regression::training::data, featuresTable);
    trainingAlgorithm.input.set(ridge_regression::training::dependentVariables, labelsTable);
    trainingAlgorithm.compute();

    // Serialize partial result
    pRes = trainingAlgorithm.getPartialResult();
    InputDataArchive dataArch;
    pRes->serialize(dataArch);
    partialResultSize = (int)dataArch.getSizeOfArchive();
    dataArch.copyArchiveToArray(partialResultPtr, (size_t)partialResultSize);

    return partialResultSize;
}

template <typename FPType>
void ridgeRegressionOnlineFinalizeTemplate(FPType * featuresPtr, FPType * labelsPtr, long long int nAllRows, int nRows, int nColumns, float l2Reg, byte * partialResultPtr, int partialResultSize,
    FPType * betaPtr, FPType * xtyPtr, FPType * xtxPtr)
{
    NumericTablePtr featuresTable(new HomogenNumericTable<FPType>(featuresPtr, nColumns, nRows));
    NumericTablePtr labelsTable(new HomogenNumericTable<FPType>(labelsPtr, 1, nRows));
    FPType l2 = l2Reg;
    NumericTablePtr l2RegTable(new HomogenNumericTable<FPType>(&l2, 1, 1));

    ridge_regression::training::Online<FPType> trainingAlgorithm;

    ridge_regression::training::PartialResultPtr pRes(new ridge_regression::training::PartialResult);
    if (partialResultSize != 0)
    {
        OutputDataArchive dataArch(partialResultPtr, partialResultSize);
        pRes->deserialize(dataArch);
        trainingAlgorithm.setPartialResult(pRes);
    }

    trainingAlgorithm.parameter.ridgeParameters = l2RegTable;

    trainingAlgorithm.input.set(ridge_regression::training::data, featuresTable);
    trainingAlgorithm.input.set(ridge_regression::training::dependentVariables, labelsTable);
    trainingAlgorithm.compute();
    trainingAlgorithm.finalizeCompute();

    ridge_regression::training::ResultPtr trainingResult = trainingAlgorithm.getResult();
    ridge_regression::ModelNormEq * model = static_cast<ridge_regression::ModelNormEq *>(trainingResult->get(ridge_regression::training::model).get());

    NumericTablePtr xtxTable = model->getXTXTable();
    const size_t nBetas = xtxTable->getNumberOfRows();
    BlockDescriptor<FPType> xtxBlock;
    xtxTable->getBlockOfRows(0, nBetas, readWrite, xtxBlock);
    FPType * xtx = xtxBlock.getBlockPtr();

    size_t offset = 0;
    for (size_t i = 0; i < nBetas; ++i)
    {
        for (size_t j = 0; j <= i; ++j)
        {
            xtxPtr[offset] = xtx[i * nBetas + j];
            offset++;
        }
    }
    offset = 0;
    for (size_t i = 0; i < nBetas; ++i)
    {
        xtxPtr[offset] += l2Reg * l2Reg * nAllRows;
        offset += i + 2;
    }

    NumericTablePtr xtyTable = model->getXTYTable();
    BlockDescriptor<FPType> xtyBlock;
    xtyTable->getBlockOfRows(0, xtyTable->getNumberOfRows(), readWrite, xtyBlock);
    FPType * xty = xtyBlock.getBlockPtr();
    for (size_t i = 0; i < nBetas; ++i)
    {
        xtyPtr[i] = xty[i];
    }

    NumericTablePtr betaTable = trainingResult->get(ridge_regression::training::model)->getBeta();
    BlockDescriptor<FPType> betaBlock;
    betaTable->getBlockOfRows(0, 1, readWrite, betaBlock);
    FPType * betaForCopy = betaBlock.getBlockPtr();
    for (size_t i = 0; i < nBetas; ++i)
    {
        betaPtr[i] = betaForCopy[i];
    }

    xtxTable->releaseBlockOfRows(xtxBlock);
    xtyTable->releaseBlockOfRows(xtyBlock);
    betaTable->releaseBlockOfRows(betaBlock);
}

EXPORT_API(int) ridgeRegressionOnlineCompute(void * featuresPtr, void * labelsPtr, int nRows, int nColumns, float l2Reg, void * partialResultPtr, int partialResultSize)
{
    return ridgeRegressionOnlineComputeTemplate<double>((double *)featuresPtr, (double *)labelsPtr, nRows, nColumns, l2Reg, (byte *)partialResultPtr, partialResultSize);
}

EXPORT_API(void) ridgeRegressionOnlineFinalize(void * featuresPtr, void * labelsPtr, long long int nAllRows, int nRows, int nColumns, float l2Reg, void * partialResultPtr, int partialResultSize,
    void * betaPtr, void * xtyPtr, void * xtxPtr)
{
    ridgeRegressionOnlineFinalizeTemplate<double>((double *)featuresPtr, (double *)labelsPtr, nAllRows, nRows, nColumns, l2Reg, (byte *)partialResultPtr, partialResultSize,
        (double *)betaPtr, (double *)xtyPtr, (double *)xtxPtr);
}

EXPORT_API(void) linearRegressionDouble(void * features, void * label, void * betas, int nRows, int nColumns)
{
    linearRegression<double>((double *)features, (double *)label, (double *)betas, nRows, nColumns);
}

EXPORT_API(void) linearRegressionSingle(void * features, void * label, void * betas, int nRows, int nColumns)
{
    linearRegression<float>((float *)features, (float *)label, (float *)betas, nRows, nColumns);
}

EXPORT_API(int) simpleTest(void * features, int nColumns)
{
    return nColumns + 42;
}

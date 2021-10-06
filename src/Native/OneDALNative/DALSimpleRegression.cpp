#include "daal.h"

#include "DALSimpleRegression.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;

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

EXPORT_API(void) linearRegressionDouble(void * features, void * label, void * betas, int nRows, int nColumns)
{
    linearRegression<double>((double *)features, (double *)label, (double *)betas, nRows, nColumns);
}

EXPORT_API(void) linearRegressionSingle(void * features, void * label, void * betas, int nRows, int nColumns)
{
    linearRegression<float>((float *)features, (float *)label, (float *)betas, nRows, nColumns);
}

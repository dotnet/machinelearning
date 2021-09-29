#include "daal.h"

#ifdef __cplusplus
#define ONEDAL_EXTERN_C extern "C"
#else
#define ONEDAL_EXTERN_C
#endif

#ifdef _MSC_VER
#define ONEDAL_EXPORT __declspec(dllexport)
#define ONEDAL_C_EXPORT ONEDAL_EXTERN_C __declspec(dllexport)
#else
#define ONEDAL_EXPORT
#define ONEDAL_C_EXPORT ONEDAL_EXTERN_C
#endif

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

ONEDAL_C_EXPORT void linearRegressionDouble(void * features, void * label, void * betas, int nRows, int nColumns)
{
    linearRegression<double>((double *)features, (double *)label, (double *)betas, nRows, nColumns);
}

ONEDAL_C_EXPORT void linearRegressionSingle(void * features, void * label, void * betas, int nRows, int nColumns)
{
    linearRegression<float>((float *)features, (float *)label, (float *)betas, nRows, nColumns);
}

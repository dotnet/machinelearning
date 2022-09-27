#include <cstdlib>
#include "daal.h"
#include "../Stdafx.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;

/*
### Logistic regression wrapper ###

public unsafe static extern void LogisticRegressionCompute(void* featuresPtr, void* labelsPtr, void* weightsPtr, bool useSampleWeights, void* betaPtr,
    long nRows, int nColumns, int nClasses, float l1Reg, float l2Reg, float accuracyThreshold, int nIterations, int m, int nThreads);
*/
template <typename FPType>
void logisticRegressionLBFGSComputeTemplate(FPType * featuresPtr, int * labelsPtr, FPType * weightsPtr, bool useSampleWeights, FPType * betaPtr,
    long long nRows, int nColumns, int nClasses, float l1Reg, float l2Reg, float accuracyThreshold, int nIterations, int m, int nThreads)
{
    bool verbose = false;
    #ifdef linux
    if (const char* env_p = std::getenv("MLNET_BACKEND_VERBOSE"))
    #elif _WIN32
    // WL Merge Note: std::getenv cause compilation error, use _dupenv_s in win, need to validate correctness.
    char * env_p;
    size_t size;
    errno_t err = _dupenv_s(&env_p, &size, "MLNET_BACKEND_VERBOSE");
    if(!err && env_p)
    #endif
    {
        verbose = true;
        printf("%s - %.12f\n", "l1Reg", l1Reg);
        printf("%s - %.12f\n", "l2Reg", l2Reg);
        printf("%s - %.12f\n", "accuracyThreshold", accuracyThreshold);
        printf("%s - %d\n", "nIterations", nIterations);
        printf("%s - %d\n", "m", m);
        printf("%s - %d\n", "nClasses", nClasses);
        printf("%s - %d\n", "nThreads", nThreads);

        const size_t nThreadsOld = Environment::getInstance()->getNumberOfThreads();
        printf("%s - %zd\n", "nThreadsOld", nThreadsOld); //Note: %lu cause compilation error, modify to %d
    }
    #ifdef _WIN32
    free(env_p);
    #endif

    Environment::getInstance()->setNumberOfThreads(nThreads);

    NumericTablePtr featuresTable(new HomogenNumericTable<FPType>(featuresPtr, nColumns, nRows));
    NumericTablePtr labelsTable(new HomogenNumericTable<int>(labelsPtr, 1, nRows));
    NumericTablePtr weightsTable(new HomogenNumericTable<FPType>(weightsPtr, 1, nRows));

    SharedPtr<optimization_solver::lbfgs::Batch<FPType>> lbfgsAlgorithm(new optimization_solver::lbfgs::Batch<FPType>());
    lbfgsAlgorithm->parameter.batchSize = featuresTable->getNumberOfRows();
    lbfgsAlgorithm->parameter.correctionPairBatchSize = featuresTable->getNumberOfRows();
    lbfgsAlgorithm->parameter.L = 1;
    lbfgsAlgorithm->parameter.m = m;
    lbfgsAlgorithm->parameter.accuracyThreshold = accuracyThreshold;
    lbfgsAlgorithm->parameter.nIterations = nIterations;

    if (nClasses == 1)
    {
        SharedPtr<optimization_solver::logistic_loss::Batch<FPType>> logLoss(new optimization_solver::logistic_loss::Batch<FPType>(featuresTable->getNumberOfRows()));
        logLoss->parameter().numberOfTerms = featuresTable->getNumberOfRows();
        logLoss->parameter().interceptFlag = true;
        logLoss->parameter().penaltyL1 = l1Reg;
        logLoss->parameter().penaltyL2 = l2Reg;

        lbfgsAlgorithm->parameter.function = logLoss;
    }
    else
    {
        SharedPtr<optimization_solver::cross_entropy_loss::Batch<FPType>> crossEntropyLoss(new optimization_solver::cross_entropy_loss::Batch<FPType>(nClasses, featuresTable->getNumberOfRows()));
        crossEntropyLoss->parameter().numberOfTerms = featuresTable->getNumberOfRows();
        crossEntropyLoss->parameter().interceptFlag = true;
        crossEntropyLoss->parameter().penaltyL1 = l1Reg;
        crossEntropyLoss->parameter().penaltyL2 = l2Reg;
        crossEntropyLoss->parameter().nClasses = nClasses;

        lbfgsAlgorithm->parameter.function = crossEntropyLoss;
    }

    logistic_regression::training::Batch<FPType> trainingAlgorithm(nClasses == 1 ? 2 : nClasses);
    trainingAlgorithm.parameter().optimizationSolver = lbfgsAlgorithm;
    trainingAlgorithm.parameter().penaltyL1 = l1Reg;
    trainingAlgorithm.parameter().penaltyL2 = l2Reg;
    trainingAlgorithm.parameter().interceptFlag = true;

    trainingAlgorithm.input.set(classifier::training::data, featuresTable);
    trainingAlgorithm.input.set(classifier::training::labels, labelsTable);
    if (useSampleWeights)
    {
        trainingAlgorithm.input.set(classifier::training::weights, weightsTable);
    }

    trainingAlgorithm.compute();

    logistic_regression::training::ResultPtr trainingResult = trainingAlgorithm.getResult();
    logistic_regression::ModelPtr modelPtr = trainingResult->get(classifier::training::model);

    NumericTablePtr betaTable = modelPtr->getBeta();
    if (betaTable->getNumberOfRows() != nClasses)
    {
        printf("Wrong number of classes in beta table\n");
    }
    if (betaTable->getNumberOfColumns() != nColumns + 1)
    {
        printf("Wrong number of features in beta table\n");
    }

    BlockDescriptor<FPType> betaBlock;
    betaTable->getBlockOfRows(0, betaTable->getNumberOfRows(), readWrite, betaBlock);
    FPType * betaForCopy = betaBlock.getBlockPtr();
    for (size_t i = 0; i < nClasses; ++i)
    {
        betaPtr[i] = betaForCopy[i * (nColumns + 1)];
    }
    for (size_t i = 0; i < nClasses; ++i)
    {
        for (size_t j = 1; j < nColumns + 1; ++j)
        {
            betaPtr[nClasses + i * nColumns + j - 1] = betaForCopy[i * (nColumns + 1) + j];
        }
    }

    if (verbose)
    {
        optimization_solver::iterative_solver::ResultPtr solverResult = lbfgsAlgorithm->getResult();
        NumericTablePtr nIterationsTable = solverResult->get(optimization_solver::iterative_solver::nIterations);
        BlockDescriptor<int> nIterationsBlock;
        nIterationsTable->getBlockOfRows(0, 1, readWrite, nIterationsBlock);
        int * nIterationsPtr = nIterationsBlock.getBlockPtr();

        printf("Solver iterations: %d\n", nIterationsPtr[0]);

        logistic_regression::prediction::Batch<FPType> predictionAlgorithm(nClasses == 1 ? 2 : nClasses);
        // predictionAlgorithm.parameter().resultsToEvaluate |=
        //     static_cast<DAAL_UINT64>(classifier::computeClassProbabilities);
        predictionAlgorithm.input.set(classifier::prediction::data, featuresTable);
        predictionAlgorithm.input.set(classifier::prediction::model, modelPtr);
        predictionAlgorithm.compute();
        NumericTablePtr predictionsTable = predictionAlgorithm.getResult()->get(classifier::prediction::prediction);
        BlockDescriptor<int> predictionsBlock;
        predictionsTable->getBlockOfRows(0, nRows, readWrite, predictionsBlock);
        int * predictions = predictionsBlock.getBlockPtr();
        FPType accuracy = 0;
        for (long i = 0; i < nRows; ++i)
        {
            if (predictions[i] == labelsPtr[i])
            {
                accuracy += 1.0;
            }
        }
        accuracy /= nRows;
        predictionsTable->releaseBlockOfRows(predictionsBlock);
        nIterationsTable->releaseBlockOfRows(nIterationsBlock);
        printf("oneDAL LogReg traning accuracy: %f\n", accuracy);
    }

    betaTable->releaseBlockOfRows(betaBlock);

}

EXPORT_API(void) logisticRegressionLBFGSCompute(void * featuresPtr, void * labelsPtr, void * weightsPtr, bool useSampleWeights, void * betaPtr,
    long long nRows, int nColumns, int nClasses, float l1Reg, float l2Reg, float accuracyThreshold, int nIterations, int m, int nThreads)
{
    return logisticRegressionLBFGSComputeTemplate<float>((float *)featuresPtr, (int *)labelsPtr, (float *)weightsPtr, useSampleWeights, (float *)betaPtr,
        nRows, nColumns, nClasses, l1Reg, l2Reg, accuracyThreshold, nIterations, m, nThreads);
}

/*
### Ridge regression wrapper ###

[DllImport(OneDalLibPath, EntryPoint = "ridgeRegressionOnlineCompute")]
public unsafe static extern int RidgeRegressionOnlineCompute(void* featuresPtr, void* labelsPtr, int nRows, int nColumns, float l2Reg, void* partialResultPtr, int partialResultSize);

[DllImport(OneDalLibPath, EntryPoint = "ridgeRegressionOnlineFinalize")]
public unsafe static extern void RidgeRegressionOnlineFinalize(void* featuresPtr, void* labelsPtr, long nAllRows, int nRows, int nColumns, float l2Reg, void* partialResultPtr, int partialResultSize,
    void* betaPtr, void* xtyPtr, void* xtxPtr);
*/
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
    const size_t nBeta = xtxTable->getNumberOfRows();
    BlockDescriptor<FPType> xtxBlock;
    xtxTable->getBlockOfRows(0, nBeta, readWrite, xtxBlock);
    FPType * xtx = xtxBlock.getBlockPtr();

    size_t offset = 0;
    for (size_t i = 0; i < nBeta; ++i)
    {
        for (size_t j = 0; j <= i; ++j)
        {
            xtxPtr[offset] = xtx[i * nBeta + j];
            offset++;
        }
    }
    offset = 0;
    for (size_t i = 0; i < nBeta; ++i)
    {
        xtxPtr[offset] += l2Reg * l2Reg * nAllRows;
        offset += i + 2;
    }

    NumericTablePtr xtyTable = model->getXTYTable();
    BlockDescriptor<FPType> xtyBlock;
    xtyTable->getBlockOfRows(0, xtyTable->getNumberOfRows(), readWrite, xtyBlock);
    FPType * xty = xtyBlock.getBlockPtr();
    for (size_t i = 0; i < nBeta; ++i)
    {
        xtyPtr[i] = xty[i];
    }

    NumericTablePtr betaTable = trainingResult->get(ridge_regression::training::model)->getBeta();
    BlockDescriptor<FPType> betaBlock;
    betaTable->getBlockOfRows(0, 1, readWrite, betaBlock);
    FPType * betaForCopy = betaBlock.getBlockPtr();
    for (size_t i = 0; i < nBeta; ++i)
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

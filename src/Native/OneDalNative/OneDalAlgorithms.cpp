#include <cstdlib>
#include <iostream>
#include "daal.h"
#include "../Stdafx.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;


bool getVerboseVariable()
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
    #else
    if(false)
    #endif
        verbose = true;
    #ifdef _WIN32
    free(env_p);
    #endif

    return verbose;
}
/*
    Decision Forest regression tree traveler
*/
template <typename FPType>
class RegressorNodeVisitor : public daal::algorithms::tree_utils::regression::TreeNodeVisitor
{
public:
    RegressorNodeVisitor(size_t numberOfLeaves, bool verbose)
    {
        _verbose = verbose;
        _numberOfLeaves = numberOfLeaves;
        _lteChild = new int[_numberOfLeaves - 1];
        _gtChild = new int[_numberOfLeaves - 1];
        _splitFeature = new int[_numberOfLeaves - 1];
        _featureThreshold = new FPType[_numberOfLeaves - 1];
        _leafValues = new FPType[_numberOfLeaves];

        _currentNode = 0;
        _currentLeaf = -1;
        _previousLevel = 0;
        _previousNodes = new size_t[1024];
    }

    virtual bool onLeafNode(const tree_utils::regression::LeafNodeDescriptor & desc)
    {
        // step down
        if (desc.level == _previousLevel + 1)
        {
            _lteChild[_previousNodes[_previousLevel]] = _currentLeaf;
        }
        // switch to different branch
        else
        {
            _gtChild[_previousNodes[desc.level - 1]] = _currentLeaf;
        }
        _leafValues[-_currentLeaf - 1] = (FPType)desc.response;
        _previousLevel = desc.level;
        _currentLeaf--;

        if (_verbose)
        {
            for (size_t i = 0; i < desc.level; ++i) std::cout << "  ";
            std::cout << "Level " << desc.level << ", leaf node. Response value = " << desc.response << ", Impurity = " << desc.impurity
                      << ", Number of samples = " << desc.nNodeSampleCount << std::endl;

            for (size_t i = 0; i < desc.level; ++i) std::cout << "  ";
            std::cout << "DEBUG: current level " << _previousLevel << " currentLeaf " << _currentLeaf + 1 << " previousNodes";
        }

        return true;
    }

    virtual bool onSplitNode(const tree_utils::regression::SplitNodeDescriptor & desc)
    {
        // step down or root node
        if (desc.level == _previousLevel + 1 || desc.level == 0)
        {
            if (desc.level != 0)
                _lteChild[_previousNodes[_previousLevel]] = _currentNode;
        }
        // switch to different branch
        else
        {
            _gtChild[_previousNodes[desc.level - 1]] = _currentNode;
        }
        _splitFeature[_currentNode] = desc.featureIndex;
        _featureThreshold[_currentNode] = (FPType)desc.featureValue;
        _previousNodes[desc.level] = _currentNode;
        _previousLevel = desc.level;
        _currentNode++;

        if (_verbose)
        {
            for (size_t i = 0; i < desc.level; ++i) std::cout << "  ";
            std::cout << "Level " << desc.level << ", split node. Feature index = " << desc.featureIndex << ", feature value = " << desc.featureValue
                      << ", Impurity = " << desc.impurity << ", Number of samples = " << desc.nNodeSampleCount << std::endl;

            for (size_t i = 0; i < desc.level; ++i) std::cout << "  ";
            std::cout << "DEBUG: current level " << _previousLevel << " currentNode " << _currentNode - 1 << " previousNodes";
            for (size_t i = 0; i < desc.level; ++i)
                std::cout << " " << _previousNodes[i];
            std::cout << std::endl;
        }

        return true;
    }

    void copyTreeStructureToBuffers(int * lteChild, int * gtChild, int * splitFeature, FPType * featureThreshold, FPType * leafValues)
    {
        for (size_t i = 0; i < _numberOfLeaves - 1; ++i)
        {
            lteChild[i] = _lteChild[i];
            gtChild[i] = _gtChild[i];
            splitFeature[i] = _splitFeature[i];
            featureThreshold[i] = _featureThreshold[i];
            leafValues[i] = _leafValues[i];
        }
        leafValues[_numberOfLeaves - 1] = _leafValues[_numberOfLeaves - 1];

        if (_verbose)
        {
            printf("Number of leaves: %d\n", -_currentLeaf - 1);
            printf("Number of nodes: %lu\n", _currentNode);
        }
    }

    ~RegressorNodeVisitor()
    {
        delete[] _previousNodes;
        delete[] _lteChild;
        delete[] _gtChild;
        delete[] _splitFeature;
        delete[] _featureThreshold;
        delete[] _leafValues;
    }

    size_t _numberOfLeaves;
    int * _lteChild;
    int * _gtChild;
    int * _splitFeature;
    FPType * _featureThreshold;
    FPType * _leafValues;

    size_t * _previousNodes;
    size_t _currentNode;
    int _currentLeaf;
    size_t _previousLevel;
    bool _verbose;
};

/*
    ### Decision Forest regression wrappers ###
    [DllImport(OneDalLibPath, EntryPoint = "decisionForestRegressionCompute")]
    public static extern unsafe int DecisionForestRegressionCompute(
        void* featuresPtr, void* labelsPtr, long nRows, int nColumns, int numberOfThreads,
        float featureFractionPerSplit, int numberOfTrees, int numberOfLeaves, int minimumExampleCountPerLeaf, int maxBins,
        void* lteChildPtr, void* gtChildPtr, void* splitFeaturePtr, void* featureThresholdPtr, void* leafValuesPtr, void* modelPtr)
*/
template <typename FPType>
int decisionForestRegressionComputeTemplate(
    FPType * featuresPtr, FPType * labelsPtr, long long nRows, int nColumns,
    int numberOfThreads, float featureFractionPerSplit, int numberOfTrees, int numberOfLeaves, int minimumExampleCountPerLeaf, int maxBins,
    int * lteChildPtr, int * gtChildPtr, int * splitFeaturePtr, FPType * featureThresholdPtr, FPType * leafValuesPtr, byte* modelPtr)
{
    bool verbose = getVerboseVariable();
    if (verbose)
    {
        printf("%s\n", "Decision Forest Regression parameters:");
        printf("\t%s - %d\n", "numberOfThreads", numberOfThreads);
        printf("\t%s - %d\n", "numberOfTrees", numberOfTrees);
        printf("\t%s - %.6f\n", "featureFractionPerSplit", featureFractionPerSplit);
        printf("\t%s - %d\n", "featureFractionPerSplit(int)", (int)(nColumns * featureFractionPerSplit));
        printf("\t%s - %d\n", "numberOfLeaves", numberOfLeaves);
        printf("\t%s - %d\n", "minimumExampleCountPerLeaf", minimumExampleCountPerLeaf);
        printf("\t%s - %d\n", "maxBins", maxBins);
    }

    if (numberOfThreads != 0)
        Environment::getInstance()->setNumberOfThreads(numberOfThreads);

    NumericTablePtr featuresTable(new HomogenNumericTable<FPType>(featuresPtr, nColumns, nRows));
    NumericTablePtr labelsTable(new HomogenNumericTable<FPType>(labelsPtr, 1, nRows));

    decision_forest::regression::training::Batch<FPType, decision_forest::regression::training::hist> algorithm;

    algorithm.input.set(decision_forest::regression::training::data, featuresTable);
    algorithm.input.set(decision_forest::regression::training::dependentVariable, labelsTable);

    algorithm.parameter().nTrees                         = numberOfTrees;
    algorithm.parameter().observationsPerTreeFraction    = 1;
    algorithm.parameter().featuresPerNode                = (int)(nColumns * featureFractionPerSplit);
    algorithm.parameter().maxTreeDepth                   = 0; // unlimited growth in depth
    algorithm.parameter().impurityThreshold              = 0;
    algorithm.parameter().varImportance                  = algorithms::decision_forest::training::MDI;
    algorithm.parameter().resultsToCompute               = algorithms::decision_forest::training::computeOutOfBagError;
    algorithm.parameter().bootstrap                      = true;
    algorithm.parameter().minObservationsInLeafNode      = minimumExampleCountPerLeaf;
    algorithm.parameter().minObservationsInSplitNode     = 2;
    algorithm.parameter().minWeightFractionInLeafNode    = 0;
    algorithm.parameter().minImpurityDecreaseInSplitNode = 0;
    algorithm.parameter().maxLeafNodes                   = numberOfLeaves;
    algorithm.parameter().maxBins                        = maxBins;
    algorithm.parameter().minBinSize                     = 5;

    algorithm.compute();

    decision_forest::regression::training::ResultPtr trainingResult = algorithm.getResult();
    decision_forest::regression::ModelPtr model = trainingResult->get(decision_forest::regression::training::model);

    InputDataArchive dataArch;
    trainingResult->serialize(dataArch);
    int modelSize = dataArch.getSizeOfArchive();
    dataArch.copyArchiveToArray(modelPtr, modelSize);

    for (size_t i = 0; i < numberOfTrees; ++i)
    {
        RegressorNodeVisitor<FPType> visitor(numberOfLeaves, verbose);
        model->traverseDFS(i, visitor);

        visitor.copyTreeStructureToBuffers(
            lteChildPtr + i * (numberOfLeaves - 1),
            gtChildPtr + i * (numberOfLeaves - 1),
            splitFeaturePtr + i * (numberOfLeaves - 1),
            featureThresholdPtr + i * (numberOfLeaves - 1),
            leafValuesPtr + i * numberOfLeaves
        );

        if (verbose)
        {
            printf("lteChild:\n");
            for (size_t j = 0; j < numberOfLeaves - 1; ++j)
                printf("%d ", lteChildPtr[i * (numberOfLeaves - 1) + j]);
            printf("\n");

            printf("gtChild:\n");
            for (size_t j = 0; j < numberOfLeaves - 1; ++j)
                printf("%d ", gtChildPtr[i * (numberOfLeaves - 1) + j]);
            printf("\n");

            printf("splitFeature:\n");
            for (size_t j = 0; j < numberOfLeaves - 1; ++j)
                printf("%d ", splitFeaturePtr[i * (numberOfLeaves - 1) + j]);
            printf("\n");

            printf("featureThreshold:\n");
            for (size_t j = 0; j < numberOfLeaves - 1; ++j)
                printf("%f ", featureThresholdPtr[i * (numberOfLeaves - 1) + j]);
            printf("\n");

            printf("leafValues:\n");
            for (size_t j = 0; j < numberOfLeaves; ++j)
                printf("%f ", leafValuesPtr[i * numberOfLeaves + j]);
            printf("\n");
        }
    }

    return modelSize;
}

EXPORT_API(int) decisionForestRegressionCompute(
    void * featuresPtr, void * labelsPtr, long long nRows, int nColumns,
    int numberOfThreads, float featureFractionPerSplit, int numberOfTrees, int numberOfLeaves, int minimumExampleCountPerLeaf, int maxBins,
    void * lteChildPtr, void * gtChildPtr, void * splitFeaturePtr, void * featureThresholdPtr, void * leafValuesPtr, void* modelPtr)
{
    return decisionForestRegressionComputeTemplate<float>(
        (float *)featuresPtr, (float *)labelsPtr, nRows, nColumns,
        numberOfThreads, featureFractionPerSplit, numberOfTrees, numberOfLeaves, minimumExampleCountPerLeaf, maxBins,
        (int *)lteChildPtr, (int *)gtChildPtr, (int *)splitFeaturePtr, (float *)featureThresholdPtr, (float *)leafValuesPtr, (byte *)modelPtr);
}

/*
    Decision Forest classification tree traveler
*/
template <typename FPType>
class ClassifierNodeVisitor : public daal::algorithms::tree_utils::classification::TreeNodeVisitor
{
public:
    ClassifierNodeVisitor(size_t numberOfLeaves, bool verbose)
    {
        _verbose = verbose;
        _numberOfLeaves = numberOfLeaves;
        _lteChild = new int[_numberOfLeaves - 1];
        _gtChild = new int[_numberOfLeaves - 1];
        _splitFeature = new int[_numberOfLeaves - 1];
        _featureThreshold = new FPType[_numberOfLeaves - 1];
        _leafValues = new FPType[_numberOfLeaves];

        _currentNode = 0;
        _currentLeaf = -1;
        _previousLevel = 0;
        _previousNodes = new size_t[1024];
    }

    virtual bool onLeafNode(const tree_utils::classification::LeafNodeDescriptor & desc)
    {
        // step down
        if (desc.level == _previousLevel + 1)
        {
            _lteChild[_previousNodes[_previousLevel]] = _currentLeaf;
        }
        // switch to different branch
        else
        {
            _gtChild[_previousNodes[desc.level - 1]] = _currentLeaf;
        }
        _leafValues[-_currentLeaf - 1] = 1 - 2 * (FPType)desc.prob[0];
        _previousLevel = desc.level;
        _currentLeaf--;

        if (_verbose)
        {
            for (size_t i = 0; i < desc.level; ++i) std::cout << "  ";
            std::cout << "Level " << desc.level << ", leaf node. Label value = " << desc.label << ", Impurity = " << desc.impurity
                      << ", Number of samples = " << desc.nNodeSampleCount << ", Probabilities = { ";
            for (size_t indexClass = 0; indexClass < 2; ++indexClass)
            {
                std::cout << desc.prob[indexClass] << ' ';
            }
            std::cout << "}" << std::endl;

            for (size_t i = 0; i < desc.level; ++i) std::cout << "  ";
            std::cout << "DEBUG: current level " << _previousLevel << " currentLeaf " << _currentLeaf + 1 << " previousNodes";
            for (size_t i = 0; i < desc.level; ++i)
                std::cout << " " << _previousNodes[i];
            std::cout << std::endl;
        }
        return true;
    }

    virtual bool onSplitNode(const tree_utils::classification::SplitNodeDescriptor & desc)
    {
        // step down or root node
        if (desc.level == _previousLevel + 1 || desc.level == 0)
        {
            if (desc.level != 0)
                _lteChild[_previousNodes[_previousLevel]] = _currentNode;
        }
        // switch to different branch
        else
        {
            _gtChild[_previousNodes[desc.level - 1]] = _currentNode;
        }
        _splitFeature[_currentNode] = desc.featureIndex;
        _featureThreshold[_currentNode] = (FPType)desc.featureValue;
        _previousNodes[desc.level] = _currentNode;
        _previousLevel = desc.level;
        _currentNode++;

        if (_verbose)
        {
            for (size_t i = 0; i < desc.level; ++i) std::cout << "  ";
            std::cout << "Level " << desc.level << ", split node. Feature index = " << desc.featureIndex << ", feature value = " << desc.featureValue
                      << ", Impurity = " << desc.impurity << ", Number of samples = " << desc.nNodeSampleCount << std::endl;

            for (size_t i = 0; i < desc.level; ++i) std::cout << "  ";
            std::cout << "DEBUG: current level " << _previousLevel << " currentNode " << _currentNode - 1 << " previousNodes";
            for (size_t i = 0; i < desc.level; ++i)
                std::cout << " " << _previousNodes[i];
            std::cout << std::endl;
        }
        return true;
    }

    void copyTreeStructureToBuffers(int * lteChild, int * gtChild, int * splitFeature, FPType * featureThreshold, FPType * leafValues)
    {
        for (size_t i = 0; i < _numberOfLeaves - 1; ++i)
        {
            lteChild[i] = _lteChild[i];
            gtChild[i] = _gtChild[i];
            splitFeature[i] = _splitFeature[i];
            featureThreshold[i] = _featureThreshold[i];
            leafValues[i] = _leafValues[i];
        }
        leafValues[_numberOfLeaves - 1] = _leafValues[_numberOfLeaves - 1];

        if (_verbose)
        {
            printf("Number of leaves: %d\n", -_currentLeaf - 1);
            printf("Number of nodes: %lu\n", _currentNode);
        }
    }

    ~ClassifierNodeVisitor()
    {
        delete[] _previousNodes;
        delete[] _lteChild;
        delete[] _gtChild;
        delete[] _splitFeature;
        delete[] _featureThreshold;
        delete[] _leafValues;
    }

    size_t _numberOfLeaves;
    int * _lteChild;
    int * _gtChild;
    int * _splitFeature;
    FPType * _featureThreshold;
    FPType * _leafValues;

    size_t * _previousNodes;
    size_t _currentNode;
    int _currentLeaf;
    size_t _previousLevel;
    bool _verbose;
};

/*
    ### Decision Forest classification wrappers ###
    [DllImport(OneDalLibPath, EntryPoint = "decisionForestClassificationCompute")]
    public static extern unsafe int DecisionForestClassificationCompute(
        void* featuresPtr, void* labelsPtr, long nRows, int nColumns, int nClasses, int numberOfThreads,
        float featureFractionPerSplit, int numberOfTrees, int numberOfLeaves, int minimumExampleCountPerLeaf, int maxBins,
        void* lteChildPtr, void* gtChildPtr, void* splitFeaturePtr, void* featureThresholdPtr, void* leafValuesPtr, void* modelPtr)
*/
template <typename FPType>
int decisionForestClassificationComputeTemplate(
    FPType * featuresPtr, FPType * labelsPtr, long long nRows, int nColumns, int nClasses,
    int numberOfThreads, float featureFractionPerSplit, int numberOfTrees, int numberOfLeaves, int minimumExampleCountPerLeaf, int maxBins,
    int * lteChildPtr, int * gtChildPtr, int * splitFeaturePtr, FPType * featureThresholdPtr, FPType * leafValuesPtr, byte* modelPtr)
{
    bool verbose = getVerboseVariable();
    if (verbose)
    {
        printf("%s\n", "Decision Forest Classification parameters:");
        printf("\t%s - %d\n", "numberOfThreads", numberOfThreads);
        printf("\t%s - %d\n", "numberOfTrees", numberOfTrees);
        printf("\t%s - %.6f\n", "featureFractionPerSplit", featureFractionPerSplit);
        printf("\t%s - %d\n", "featureFractionPerSplit(int)", (int)(nColumns * featureFractionPerSplit));
        printf("\t%s - %d\n", "numberOfLeaves", numberOfLeaves);
        printf("\t%s - %d\n", "minimumExampleCountPerLeaf", minimumExampleCountPerLeaf);
        printf("\t%s - %d\n", "maxBins", maxBins);
    }

    if (numberOfThreads != 0)
        Environment::getInstance()->setNumberOfThreads(numberOfThreads);

    NumericTablePtr featuresTable(new HomogenNumericTable<FPType>(featuresPtr, nColumns, nRows));
    NumericTablePtr labelsTable(new HomogenNumericTable<FPType>(labelsPtr, 1, nRows));

    decision_forest::classification::training::Batch<FPType, decision_forest::classification::training::hist> algorithm(nClasses);

    algorithm.input.set(classifier::training::data, featuresTable);
    algorithm.input.set(classifier::training::labels, labelsTable);

    algorithm.parameter().nTrees                         = numberOfTrees;
    algorithm.parameter().observationsPerTreeFraction    = 1;
    algorithm.parameter().featuresPerNode                = (int)(nColumns * featureFractionPerSplit);
    algorithm.parameter().maxTreeDepth                   = 0; // unlimited growth in depth
    algorithm.parameter().impurityThreshold              = 0;
    algorithm.parameter().varImportance                  = algorithms::decision_forest::training::MDI;
    algorithm.parameter().resultsToCompute               = algorithms::decision_forest::training::computeOutOfBagError;
    algorithm.parameter().bootstrap                      = true;
    algorithm.parameter().minObservationsInLeafNode      = minimumExampleCountPerLeaf;
    algorithm.parameter().minObservationsInSplitNode     = 2;
    algorithm.parameter().minWeightFractionInLeafNode    = 0;
    algorithm.parameter().minImpurityDecreaseInSplitNode = 0;
    algorithm.parameter().maxLeafNodes                   = numberOfLeaves;
    algorithm.parameter().maxBins                        = maxBins;
    algorithm.parameter().minBinSize                     = 5;

    algorithm.compute();

    decision_forest::classification::training::ResultPtr trainingResult = algorithm.getResult();
    decision_forest::classification::ModelPtr model = trainingResult->get(classifier::training::model);

    InputDataArchive dataArch;
    trainingResult->serialize(dataArch);
    int modelSize = dataArch.getSizeOfArchive();
    dataArch.copyArchiveToArray(modelPtr, modelSize);

    for (size_t i = 0; i < numberOfTrees; ++i)
    {
        ClassifierNodeVisitor<FPType> visitor(numberOfLeaves, verbose);
        model->traverseDFS(i, visitor);

        visitor.copyTreeStructureToBuffers(
            lteChildPtr + i * (numberOfLeaves - 1),
            gtChildPtr + i * (numberOfLeaves - 1),
            splitFeaturePtr + i * (numberOfLeaves - 1),
            featureThresholdPtr + i * (numberOfLeaves - 1),
            leafValuesPtr + i * numberOfLeaves
        );

        if (verbose)
        {
            printf("lteChild:\n");
            for (size_t j = 0; j < numberOfLeaves - 1; ++j)
                printf("%d ", lteChildPtr[i * (numberOfLeaves - 1) + j]);
            printf("\n");

            printf("gtChild:\n");
            for (size_t j = 0; j < numberOfLeaves - 1; ++j)
                printf("%d ", gtChildPtr[i * (numberOfLeaves - 1) + j]);
            printf("\n");

            printf("splitFeature:\n");
            for (size_t j = 0; j < numberOfLeaves - 1; ++j)
                printf("%d ", splitFeaturePtr[i * (numberOfLeaves - 1) + j]);
            printf("\n");

            printf("featureThreshold:\n");
            for (size_t j = 0; j < numberOfLeaves - 1; ++j)
                printf("%f ", featureThresholdPtr[i * (numberOfLeaves - 1) + j]);
            printf("\n");

            printf("leafValues:\n");
            for (size_t j = 0; j < numberOfLeaves; ++j)
                printf("%f ", leafValuesPtr[i * numberOfLeaves + j]);
            printf("\n");
        }
    }

    return modelSize;
}

EXPORT_API(int) decisionForestClassificationCompute(
    void * featuresPtr, void * labelsPtr, long long nRows, int nColumns, int nClasses,
    int numberOfThreads, float featureFractionPerSplit, int numberOfTrees, int numberOfLeaves, int minimumExampleCountPerLeaf, int maxBins,
    void * lteChildPtr, void * gtChildPtr, void * splitFeaturePtr, void * featureThresholdPtr, void * leafValuesPtr, void* modelPtr)
{
    return decisionForestClassificationComputeTemplate<float>(
        (float *)featuresPtr, (float *)labelsPtr, nRows, nColumns, nClasses,
        numberOfThreads, featureFractionPerSplit, numberOfTrees, numberOfLeaves, minimumExampleCountPerLeaf, maxBins,
        (int *)lteChildPtr, (int *)gtChildPtr, (int *)splitFeaturePtr, (float *)featureThresholdPtr, (float *)leafValuesPtr, (byte *)modelPtr);
}
/*
    ### Logistic regression wrapper ###
    public unsafe static extern void LogisticRegressionCompute(void* featuresPtr, void* labelsPtr, void* weightsPtr, bool useSampleWeights, void* betaPtr,
        long nRows, int nColumns, int nClasses, float l1Reg, float l2Reg, float accuracyThreshold, int nIterations, int m, int nThreads);
*/
template <typename FPType>
void logisticRegressionLBFGSComputeTemplate(FPType * featuresPtr, int * labelsPtr, FPType * weightsPtr, bool useSampleWeights, FPType * betaPtr,
    long long nRows, int nColumns, int nClasses, float l1Reg, float l2Reg, float accuracyThreshold, int nIterations, int m, int nThreads)
{
    bool verbose = getVerboseVariable();
    if (verbose)
    {
        printf("%s\n", "Logistic Regression parameters:");
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

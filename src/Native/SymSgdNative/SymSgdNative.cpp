// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// SymSGDNative.cpp : Defines the exported functions for the DLL application.

#include <vector>
#include <random>
#if defined(USE_OMP)  
#include <omp.h>
#endif
#include <unordered_map>
#include "../Stdafx.h"
#include "Macros.h"
#include "SparseBLAS.h"
#include "SymSgdNative.h"

// This method learns for a single instance
inline void LearnInstance(int instSize, int* instIndices, float* instValues,
    float label, float alpha, float l2Const, float piw, float& weightScaling, float* weightVector, float& bias)
{
    float dotProduct = 0.0f;
    if (instIndices) // If it is a sparse instance
        dotProduct = SDOTI(instSize, instIndices, instValues, weightVector) * weightScaling + bias;
    else // If it is dense case.
        dotProduct = SDOT(instSize, instValues, weightVector)*weightScaling + bias;
    // Compute the derivative coefficient
    float sigmoidPrediction = 1.0f / (1.0f + exp(-dotProduct));
    float derivative = (label > 0) ? piw * (sigmoidPrediction - 1) : sigmoidPrediction;
    float derivativeCoef = -alpha * derivative;
    weightScaling *= (1.0f - alpha*l2Const);
    // Apply the derivative back to the weightVector
    if (instIndices) // If it is a sparse instance
        SAXPYI(instSize, instIndices, instValues, weightVector, derivativeCoef / weightScaling);
    else
        SAXPY(instSize, instValues, weightVector, derivativeCoef / weightScaling);
    
    bias = bias + derivativeCoef;
}

// This method permutes frequent features with starting consecutive features
void ComputeRemapping(int totalNumInstances, int* instSizes, int** instIndices,
    int numFeat, int numLocIter, int numThreads, SymSGDState* state, int& numFreqFeat)
{
    // There are two maps used for this permutation: 
    // 1) a direct map
    state->FreqFeatDirectMap = new int[numFeat];
    // 2) an unordered map
    state->FreqFeatUnorderedMap = new std::unordered_map<int, int>();

    int* freqFeatDirectMap = state->FreqFeatDirectMap;
    std::unordered_map<int, int>* freqFeatUnorderedMap = (std::unordered_map<int, int>*)state->FreqFeatUnorderedMap;

    // If numLocIter is 1, it means that every iteration must be reduced and therefore, no feature should be considered frequent
    // In this special case, we do not need a mapping since it is identity.
    if (numLocIter == 1)
    {
        memset(freqFeatDirectMap, -1, sizeof(int)*numFeat);
        numFreqFeat = 0;
        return;
    }

    memset(freqFeatDirectMap, 0, sizeof(int)*numFeat);

    // Frequent features are searched by subSampling the data and histogramming the frequecy of each feature
    int subSampleSize = MIN(1000 * numLocIter*numThreads, totalNumInstances);
    // The threshold to call a feature frequent
    float threshold = (float)subSampleSize / (float)numLocIter;

    // Compute the histogram for the frequency of features
    // freqFeatDirectMap is used to store the histogram
    for (int i = 0; i < subSampleSize; i++)
    {
        if (instIndices[i])
        {
            for (int j = 0; j < instSizes[i]; j++) {
                freqFeatDirectMap[instIndices[i][j]]++;
            }
        } else
        {
            for (int j = 0; j < numFeat; j++) {
                freqFeatDirectMap[j]++;
            }
        }
    }
    // Compute the permutation that is required to re-order features such that
    // frequent features are at the beginning of feature space

    // feature i and feautre numFreqFeat are subject to swap. The only difficulty is when
    // numFreqFeat is already recognized as a frequent feature.
    // Variable numFreqFeat keeps the numFreqFeat observed so far
    numFreqFeat = 0;
    for (int i = 0; i < numFeat; i++)
    {
        // Check if i is a frequent feature
        if (freqFeatDirectMap[i] > threshold)
        {
            // Check if all the features seen so far are all frequent
            if (numFreqFeat != i)
            {
                // We have to swap i with numFreqFeat
                auto searchedRes = freqFeatUnorderedMap->find(numFreqFeat);
                // Check if numFreqFeat is already occupied in freqFeatureUnorderedMap 
                // which means that numFreqFeat is already a frequent feature
                if (searchedRes == freqFeatUnorderedMap->end())
                {
                    // In this case, numFreqFeature is unoccupied and can be used for permutation for i.
                    (*freqFeatUnorderedMap)[i] = numFreqFeat;
                    (*freqFeatUnorderedMap)[numFreqFeat] = i;
                } else
                {
                    // Since numFreqFeat is already a frequent feature, its mapped index, (*freqFeatUnorderedMap)[numFreqFeat],
                    // was a non-frequent feature. So in this case, numFreqFeat mapping is removed and its mapped feature index 
                    // is used for i instead.
                    int oldFreqFeat = numFreqFeat;
                    int oldFreqFeatMappedTo = searchedRes->second;
                    freqFeatUnorderedMap->erase(searchedRes);
                    (*freqFeatUnorderedMap)[oldFreqFeatMappedTo] = i;
                    (*freqFeatUnorderedMap)[i] = oldFreqFeatMappedTo;
                }
            }
            numFreqFeat++;
        }
        // freqFeatDirectMap is a direct map and -1 means freqFeatDirectMap[i] = i (identity mapping)
        freqFeatDirectMap[i] = -1;
    }

    // Here, using the unordered_map, we set the direct map accordingly
    auto endOfUnorderedMap = freqFeatUnorderedMap->end();
    for (auto it = freqFeatUnorderedMap->begin(); it != endOfUnorderedMap; it++) 
    {
        freqFeatDirectMap[it->first] = it->second;
    }
}

// This method, remap an instance using the maps provided by ComputeRemapping
void RemapInstances(int* instSizes, int** instIndices, float** instValues, int myStart, int myEnd, SymSGDState* state)
{
    int* freqFeatDirectMap = state->FreqFeatDirectMap;
    std::unordered_map<int, int>* freqFeatUnorderedMap = (std::unordered_map<int, int>*)state->FreqFeatUnorderedMap;
    auto itBegin = freqFeatUnorderedMap->begin();
    auto itEnd = freqFeatUnorderedMap->begin();
    for (int j = myStart; j < myEnd; j++) 
    {
        int instSize = instSizes[j];
        // Check if instance is sparse
        if (instIndices[j]) 
        {
            // Just swap the indices accordingly
            for (int k = 0; k < instSize; k++) 
            {
                int oldIndex = instIndices[j][k];
                // Direct map is used here since it is much more efficient to access it
                auto mappedIndex = freqFeatDirectMap[oldIndex];
                if (mappedIndex != -1) 
                {
                    int newIndex = mappedIndex;
                    instIndices[j][k] = newIndex;
                }
            }
        } else
        {
            // If the instance is dense, we have to swap the values instead
            for (auto it = itBegin; it != itEnd; it++) 
            {
                float temp = instValues[j][it->second];
                instValues[j][it->second] = instValues[j][it->first];
                instValues[j][it->first] = temp;
            }
        }
    }
}

float MaxPossibleAlpha(float alpha, float l2Const, int totalNumInstances) 
{
    return (1.0f - pow(10.0f, -6.0f / (float)totalNumInstances)) / l2Const;
}

void TuneAlpha(float& alpha, float l2Const, int totalNumInstances, int* instSizes, int** instIndices,
    float** instValues, int numFeat, int numThreads) 
{
    alpha = 1e0f;
    int logSqrtNumInst = (int) round(log10(sqrt(totalNumInstances)))-3;
    //int logAverageNorm = int(log10(averageNorm));
    if (logSqrtNumInst > 0)
        for (int i = 0; i < logSqrtNumInst; i++)
            alpha = alpha / 10.0f;
    else if (logSqrtNumInst < 0)
        for (int i = 0; i < -logSqrtNumInst; i++)
            alpha = alpha * 10.0f;

    // If we have l2Const > 0, we want to make sure alpha is not too large
    if (l2Const > 0) 
    {
        // Since weightScaling is multiplied by (1-alpha*lambda),
        // we should make sure (1-alpha*lamda)^totalNumInstances > 1e-6 which is
        // the threshold for applying the weightScaling. Therefore,
        // alpha < (1-10^(-6/totalNumInstances))/l2Const.
        alpha = MIN(alpha, MaxPossibleAlpha(alpha, l2Const, totalNumInstances));
    }

    printf("Initial learning rate is tuned to %f\n", alpha);
}


void TuneNumLocIter(int& numLocIter, int totalNumInstances, int* instSizes, int numThreads) 
{
    int averageInstSizes = 0;
    for (int i = 0; i < totalNumInstances; i++)
        averageInstSizes += instSizes[i];
    averageInstSizes = averageInstSizes / totalNumInstances;

    if (averageInstSizes > 1000)
        numLocIter = 40 / numThreads;
    else
        numLocIter = 160 / numThreads;
}


// This method sets SymSGDState and computes Remapping for indices and allocates the 
// required memory for SymSGD learners.
void InitializeState(int totalNumInstances, int* instSizes, int** instIndices, float** instValues, 
    int numFeat, bool tuneNumLocIter, int& numLocIter, int numThreads, bool tuneAlpha, float& alpha, 
    float l2Const, SymSGDState* state)
{
    if (tuneAlpha) 
    {
        TuneAlpha(alpha, l2Const, totalNumInstances, instSizes, instIndices, instValues, numFeat, numThreads);
    } else
    {
        // Check if user alpha is too large because of l2Const. Check the comment about positive l2Const in TuneAlpha.
        float maxPossibleAlpha = MaxPossibleAlpha(alpha, l2Const, totalNumInstances);
        if (alpha > maxPossibleAlpha)
            printf("Warning: learning rate is too high! Try using a value < %e instead\n", maxPossibleAlpha);
    }

    if (tuneNumLocIter)
        TuneNumLocIter(numLocIter, totalNumInstances, instSizes, numThreads);

    state->WeightScaling = 1.0f;
#if defined(USE_OMP)  
    if (numThreads > 1)
    {
        state->PassIteration = 0;
        state->NumFrequentFeatures = 0;
        state->TotalInstancesProcessed = 0;
        ComputeRemapping(totalNumInstances, instSizes, instIndices, numFeat,
            numLocIter, numThreads, state, state->NumFrequentFeatures);
        printf("Number of frequent features: %d\nNumber of features: %d\n", state->NumFrequentFeatures, numFeat);

        state->NumLearners = numThreads;
        state->Learners = new SymSGD*[numThreads];
        SymSGD** learners = (SymSGD**)(state->Learners);

        // Allocation of SymSGD learners happens in parallel to follow the first touch policy.
        #pragma omp parallel num_threads(numThreads)
        {
            int threadId = omp_get_thread_num();
            learners[threadId] = new SymSGD(state->NumFrequentFeatures, threadId);
        }
    }
    
    // To make sure that MKL runs sequentially
    omp_set_num_threads(1);
#endif
}

float Loss(int instSize, int* instIndices, float* instValues,
    float label, float piw, float& weightScaling, float* weightVector, float& bias)
{
    float dotProduct = 0.0f;
    if (instIndices) // If it is a sparse instance
        dotProduct = SDOTI(instSize, instIndices, instValues, weightVector) * weightScaling + bias;
    else // If it is dense case.
        dotProduct = SDOT(instSize, instValues, weightVector) * weightScaling + bias;
    float sigmoidPrediction = 1.0f / (1.0f + exp(-dotProduct));
    float loss = (label > 0) ? -log2(sigmoidPrediction) : -log2(1 - sigmoidPrediction);
    // To prevent from loss going to infinity
    if (loss > 100.0f)
        loss = 100.0f;
    if (label > 0)
        loss *= piw;
    return loss;
}

// This methdo learns for loaded instance for as many passes as demanded
// Note that InitializeState should be called before this method
EXPORT_API(void) LearnAll(int totalNumInstances, int* instSizes, int** instIndices, float** instValues,
    float* labels, bool tuneAlpha, float& alpha, float l2Const, float piw, float* weightVector, 
    float& bias, int numFeat, int numPasses, int numThreads, bool tuneNumLocIter, int& numLocIter, float tolerance,
    bool needShuffle, bool shouldInitialize, SymSGDState* state)
{
    // If this is the first time LearnAll is called, initialize it.
    if (shouldInitialize)
        InitializeState(totalNumInstances, instSizes, instIndices, instValues, numFeat, tuneNumLocIter, numLocIter, numThreads, tuneAlpha, alpha, l2Const, state);
    float& weightScaling = state->WeightScaling;

    float totalAverageLoss = 0.0f; // Reserved for total loss computation
    float oldAverageLoss     = INFINITY; 
    float olderAverageLoss   = INFINITY;
    float oldestAverageLoss  = INFINITY;
    float totalOverallAverageLoss = INFINITY;

    float adjustedAlpha = alpha;
    // Check if totalNumInstances is too small to run in parallel
    if (numThreads == 1 || totalNumInstances < numThreads)
    {
        // For i=[0..totalNumInstances-1], (curPermMultiplier*i) % totalNumInstances always creates a pseduo random permutation
        int64_t curPermMultiplier = (VERYLARGEPRIME % totalNumInstances);
        // In the sequential case, just apply normal SGD
        for (int i = 0; i < numPasses; i++)
        {
            for (int j = 0; j < totalNumInstances; j++)
            {
                int64_t index = j;
                if (needShuffle)
                    index = (((int64_t)index * (int64_t)curPermMultiplier) % (int64_t)totalNumInstances);
                // alpha decays with the square root of number of instances processed.
                float thisAlpha = adjustedAlpha / (float)sqrt(1 + state->PassIteration * totalNumInstances + j);
                LearnInstance(instSizes[index], instIndices[index], instValues[index], labels[index], thisAlpha, l2Const, piw,
                    weightScaling, weightVector, bias);
                //state->TotalInstancesProcessed++;
                if (weightScaling < 1e-6)
                {
                    for (int k = 0; k < numFeat; k++)
                    {
                        weightVector[k] *= weightScaling;
                    }
                    weightScaling = 1.0f;
                }
            }

            float averageLoss = 0.0f;
            // Computing the total loss
            for (int j = 0; j < totalNumInstances; j++)
                averageLoss += Loss(instSizes[j], instIndices[j], instValues[j], labels[j], piw, weightScaling, weightVector, bias);
            averageLoss = averageLoss / (float)totalNumInstances;
            // If we the loss did not improve the learning rate was high, decay it.
            if (tuneAlpha && oldAverageLoss - averageLoss < tolerance)
                adjustedAlpha = adjustedAlpha / 10.0f;
            float overallAverageLoss = oldestAverageLoss - averageLoss;
            oldestAverageLoss = olderAverageLoss;
            olderAverageLoss = oldAverageLoss;
            oldAverageLoss = averageLoss;

            averageLoss = 0.0f;
            // Terminate if average loss difference between current model and the model from 3 passes ago is small
            if (overallAverageLoss < tolerance)
                break;

            // For shuffling in the next passes, instead of curPermMultiplier, use curPermMultiplier^2 which has exactly the same effect
            if (needShuffle)
                curPermMultiplier = (((int64_t)curPermMultiplier * (int64_t)curPermMultiplier) % (int64_t)totalNumInstances);
            state->PassIteration++;
        }
    } else 
    {
#if defined(USE_OMP) 
        // In parallel case...
        bool shouldRemap = !((std::unordered_map<int, int>*)state->FreqFeatUnorderedMap)->empty();
        SymSGD** learners = (SymSGD**)(state->Learners);

        float oldWeightScaling = 1.0f;
        #pragma omp parallel num_threads(numThreads)
        {
            int threadId = omp_get_thread_num();
            // Compute the portion of instances associated with threadId
            int myStart = (totalNumInstances * threadId) / numThreads;
            int myEnd = (totalNumInstances * (threadId + 1)) / numThreads;
            int myRangeLength = myEnd - myStart;

            if (shouldRemap)
                RemapInstances(instSizes, instIndices, instValues, myStart, myEnd, state);

            // This variable is used to keep track of how many instances are learned so far to do a reduction
            int instancesLearnedSinceReduction = 0;

            learners[threadId]->ResetModel(bias, weightVector, weightScaling);
            int64_t curPermMultiplier = (VERYLARGEPRIME % myRangeLength);

            for (int i = 0; i < numPasses; i++) 
            {
                for (int j = 0; j < myRangeLength; j++) 
                {
                    int64_t index = myStart + j;
                    if (needShuffle)
                        index = myStart + (((int64_t)j * (int64_t)curPermMultiplier) % (int64_t)myRangeLength);
                    // alpha decays with the square root of number of instances processed.
                    float thisAlpha = adjustedAlpha / (float)sqrt(1 + state->PassIteration*totalNumInstances + j*numThreads);
                    learners[threadId]->LearnLocalModel(instSizes[index], instIndices[index], instValues[index], labels[index],
                        thisAlpha, l2Const, piw, weightVector);
                    instancesLearnedSinceReduction++;
                    // If it reached numLocIter, do a reduction
                    if (instancesLearnedSinceReduction == numLocIter) 
                    {
                        learners[threadId]->Reduction(weightVector, bias, weightScaling);
                        learners[threadId]->ResetModel(bias, weightVector, weightScaling);
                        instancesLearnedSinceReduction = 0;
                    }
                }

                // Check if we need to reweight the weight vector
                if (l2Const > 0.0f) 
                {
                    #pragma omp barrier
                    if (weightScaling < 1e-6) 
                    {
                        #pragma omp for
                        for (int featIndex = 0; featIndex < numFeat; featIndex++)
                            weightVector[featIndex] *= weightScaling;
                        if (threadId == 0)
                            weightScaling = 1.0f;
                    }
                }

                #pragma omp barrier
                #pragma omp for reduction(+:totalAverageLoss)
                for (int j = 0; j < totalNumInstances; j++)
                    totalAverageLoss += Loss(instSizes[j], instIndices[j], instValues[j], labels[j], piw, weightScaling, weightVector, bias);
                #pragma omp barrier
                if (threadId == 0) 
                {
                    totalAverageLoss = totalAverageLoss / (float)totalNumInstances;
                    // If we the loss did not improve the learning rate was high, decay it.
                    if (tuneAlpha && oldAverageLoss - totalAverageLoss < tolerance)
                        adjustedAlpha = adjustedAlpha / 10.0f;
                    state->PassIteration++;
                    totalOverallAverageLoss = oldestAverageLoss - totalAverageLoss;
                    oldestAverageLoss = olderAverageLoss;
                    olderAverageLoss = oldAverageLoss;
                    oldAverageLoss = totalAverageLoss;

                    totalAverageLoss = 0.0f;
                }
                #pragma omp barrier
                // Terminate if average loss difference between current model and the model from 3 passes ago is small
                if (totalOverallAverageLoss < tolerance)
                    break;

                if (needShuffle)
                    curPermMultiplier = (((int64_t)curPermMultiplier * (int64_t)curPermMultiplier) % (int64_t)myRangeLength);
            }
        }
        state->TotalInstancesProcessed += numPasses*totalNumInstances;
#endif        
    }
}

// This method maps back the weight vector to the original feature space
EXPORT_API(void) MapBackWeightVector(float* weightVector, SymSGDState* state) 
{
    std::unordered_map<int, int>* freqFeatUnorderedMap = (std::unordered_map<int, int>*)state->FreqFeatUnorderedMap;
    auto endOfUnorderedMap = freqFeatUnorderedMap->end();
    for (auto it = freqFeatUnorderedMap->begin(); it != endOfUnorderedMap; it++) 
    {
        if (it->first < it->second) 
        {
            float temp = weightVector[it->second];
            weightVector[it->second] = weightVector[it->first];
            weightVector[it->first] = temp;
        }
    }
}

// Deallocation method
EXPORT_API(void) DeallocateSequentially(SymSGDState* state) 
{
#if defined(USE_OMP)     
    // To make sure that for the rest of MKL calls use parallelism
    omp_set_num_threads(omp_get_num_procs());
#endif

    SymSGD** learners = (SymSGD**)(state->Learners);
    if (learners) 
    {
        for (int i = 0; i < state->NumLearners; i++)
            delete learners[i];
    }
    if (state->FreqFeatUnorderedMap)
        delete (std::unordered_map<int, int>*)state->FreqFeatUnorderedMap;
    if (state->FreqFeatDirectMap)
        delete[] state->FreqFeatDirectMap;
}
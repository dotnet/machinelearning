#pragma once
#include "../stdafx.h"

using namespace std;

// In almost every sparse dataset, there is a great imbalance in frequency of features
// This class learns a local model for frequent features and modifies the global model for non-frequent features
// and only applies the frequent features it learned locally to the global model after certain number of iterations
class SymSGD {
private:
	int _numFreqFeat;
	// Local models that is learned
	float* _localModel;
	// A copy of the local model when it started learning
	float* _startModel;

	// Bias is by default a frequent feature
	float _bias, _startBias;

	// Local weightScaling for L2 regularization
	float _weightScaling, _startWeightScaling;
public:
	SymSGD(int numFreqFeat, int seed) {
		_numFreqFeat = numFreqFeat;
		if (numFreqFeat > 0) {
			_localModel = new float[numFreqFeat];
			_startModel = new float[numFreqFeat];
		} else
		{
			_localModel = NULL;
			_startModel = NULL;
		}
	}

	~SymSGD() {
		if (_numFreqFeat > 0) {
			delete[] _localModel;
			delete[] _startModel;
		}
	}

	// Learns for the local model on frequent features and global model for non-frequent features
	void LearnLocalModel(int instSize, int * instIndices,
		float * instValues, float instLabel, float alpha, float l2Const, float piw, float* globModel) {
		float dotProduct = 0.0f;
		// Check if it is a sparse instance
		if (instIndices) {
			for (int i = 0; i < instSize; i++) {
				int curIndex = instIndices[i];
				if (curIndex < _numFreqFeat) {
					// dotProduct on freqeunt features are computed with local model
					dotProduct += _localModel[curIndex] * instValues[i];
				} else
				{
					// Otherwise on global model
					dotProduct += globModel[curIndex] * instValues[i];
				}
			}
		} else
		{
			// In dense case scenario, there is no need to check on indices
			dotProduct += SDOT(_numFreqFeat, &_localModel[0], instValues) +
				SDOT(instSize - _numFreqFeat, &globModel[_numFreqFeat], &instValues[_numFreqFeat]);
		}
		dotProduct = dotProduct*_weightScaling + _bias;

		_weightScaling *= (1.0f - alpha*l2Const);

		// Compute the derivative coefficient
		float sigmoidPrediction = 1.0f / (1.0f + exp(-dotProduct));
		float derivative = (instLabel > 0) ? (sigmoidPrediction - 1) : sigmoidPrediction;
		if (instLabel > 0)
			derivative *= piw;
		float derivativeCoef = -2 * alpha * derivative;
		float derivativeWeightScaledCoef = derivativeCoef / _weightScaling;
		if (instIndices) {
			for (int i = 0; i < instSize; i++) {
				int curIndex = instIndices[i];
				if (curIndex < _numFreqFeat) { // Apply the gradient to the local model for frequent features
					_localModel[curIndex] += derivativeWeightScaledCoef * instValues[i];
				} else // Apply the gradient to the global model for non-frequent features
				{
					globModel[curIndex] += derivativeWeightScaledCoef * instValues[i];
				}
			}
		} else
		{
			// In dense case scenario, there is no need to check on indices
			SAXPY(_numFreqFeat, instValues, &_localModel[0], derivativeWeightScaledCoef);
			SAXPY(instSize - _numFreqFeat, &instValues[_numFreqFeat], &globModel[_numFreqFeat], derivativeWeightScaledCoef);
		}
		_bias = _bias + derivativeCoef;
	}

	// This method copies the global models to _localModel and _startModel
	void ResetModel(float bias, float* globModel, float weightScaling) {
		memcpy(&_localModel[0], globModel, _numFreqFeat * sizeof(float));
		memcpy(&_startModel[0], &_localModel[0], _numFreqFeat * sizeof(float));

		_bias = bias;
		_startBias = bias;
		_weightScaling = weightScaling;
		_startWeightScaling = weightScaling;
	}

	// Adds the delta of the _localModel and _startModel to the global model for frequent features
	void Reduction(float* globModel, float& bias, float& weightScaling) {
		for (int i = 0; i < _numFreqFeat; i++) {
			globModel[i] += _localModel[i] - _startModel[i];
		}
		bias += _bias - _startBias;
		weightScaling *= (_weightScaling / _startWeightScaling);
	}
};

// The state that is shared between SymSGD and SymSGDNative
struct SymSGDState
{
	int NumLearners;
	int TotalInstancesProcessed;
	void* Learners;
	void* FreqFeatUnorderedMap;
	int* FreqFeatDirectMap;
	int NumFrequentFeatures;
	int PassIteration;
	float WeightScaling;
};
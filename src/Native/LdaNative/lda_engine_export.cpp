// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#include "lda_engine.hpp"

/// This file use to expose public API to be consumed by TLC.
namespace lda {

    EXPORT_API(LdaEngine*) CreateEngine(int numTopic, int numVocab, float alphaSum, float beta, int numIter, int likelihoodInterval, int numThread, int mhstep, int maxDocToken)
    {
        return new LdaEngine(numTopic, numVocab, alphaSum, beta, numIter, likelihoodInterval, numThread, mhstep, maxDocToken);
    }

    EXPORT_API(void) DestroyEngine(LdaEngine* engine)
    {
        delete engine;
    }

    EXPORT_API(void) AllocateModelMemory(LdaEngine* engine, int numTopic, int numVocab, int64_t tableSize, int64_t aliasTableSize)
    {
        engine->AllocateModelMemory(numVocab, numTopic, tableSize, aliasTableSize);
    }

    EXPORT_API(void) AllocateDataMemory(LdaEngine* engine, int num_document, int64_t corpus_size)
    {
        engine->AllocateDataMemory(num_document, corpus_size);
    }

    EXPORT_API(void) Train(LdaEngine* engine, const char* trainOutput)
    {
        engine->Train(trainOutput);
    }

    EXPORT_API(void) Test(LdaEngine* engine, int32_t burnin_iter, float* pLoglikelihood)
    {
        engine->Test(burnin_iter, pLoglikelihood);
    }

    EXPORT_API(void) CleanData(LdaEngine* engine)
    {
        engine->ClearData();
    }

    EXPORT_API(void) CleanModel(LdaEngine* engine)
    {
        engine->ClearModel();
    }

    EXPORT_API(void) GetModelStat(LdaEngine* engine, int64_t &memBlockSize, int64_t &aliasMemBlockSize)
    {
        engine->GetModelStat(memBlockSize, aliasMemBlockSize);
    }

    EXPORT_API(void) GetWordTopic(LdaEngine* engine, int32_t wordId, int32_t* pTopic, int32_t* pProb, int32_t& length)
    {
        engine->GetWordTopic(wordId, pTopic, pProb, length);
    }

    EXPORT_API(void) SetWordTopic(LdaEngine* engine, int32_t wordId, int32_t* pTopic, int32_t* pProb, int32_t length)
    {
        engine->SetWordTopic(wordId, pTopic, pProb, length);
    }

    EXPORT_API(void) GetTopicSummary(LdaEngine* engine, int32_t topicId, int32_t* pWords, float* pProb, int32_t& length)
    {
        engine->GetTopicSummary(topicId, pWords, pProb, length);
    }

    EXPORT_API(void) SetAlphaSum(LdaEngine* engine, float avgDocLength)
    {
        engine->SetAlphaSum(avgDocLength);
    }

    EXPORT_API(int) FeedInData(LdaEngine* engine, int* term_id, int* term_freq, int32_t term_num, int32_t vocab_size)
    {
        return engine->FeedInData(term_id, term_freq, term_num, vocab_size);
    }

    EXPORT_API(int) FeedInDataDense(LdaEngine* engine, int* term_freq, int32_t term_num, int32_t vocab_size)
    {
        return engine->FeedInDataDense(term_freq, term_num, vocab_size);
    }

    EXPORT_API(void) GetDocTopic(LdaEngine* engine, int docID, int* pTopic, int* pProb, int32_t& numTopicReturn)
    {
        engine->GetDocTopic(docID, pTopic, pProb, numTopicReturn);
    }

    EXPORT_API(void) TestOneDoc(LdaEngine* engine, int* term_id, int* term_freq, int32_t term_num, int* pTopics, int* pProbs, int32_t& numTopicsMax, int32_t numBurnIter, bool reset)
    {
        engine->TestOneDoc(term_id, term_freq, term_num, pTopics, pProbs, numTopicsMax, numBurnIter, reset);
    }

    EXPORT_API(void) TestOneDocDense(LdaEngine* engine, int* term_freq, int32_t term_num, int* pTopics, int* pProbs, int32_t& numTopicsMax, int32_t numBurnIter, bool reset)
    {
        engine->TestOneDocDense(term_freq, term_num, pTopics, pProbs, numTopicsMax, numBurnIter, reset);
    }

    EXPORT_API(void) InitializeBeforeTrain(LdaEngine* engine)
    {
        engine->InitializeBeforeTrain();
    }

    EXPORT_API(void) InitializeBeforeTest(LdaEngine* engine)
    {
        engine->InitializeBeforeTest();
    }
}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once

#include <memory>
#include <cstdint>
#include <utility>
#include <vector>
#include <string>
#include <set>

#include "lda_document.h"
#include "hybrid_map.h"
#include "hybrid_alias_map.h"

#include "alias_multinomial_rng_int.hpp"

#ifdef _MSC_VER
#define EXPORT_API(ret) extern "C" __declspec(dllexport) ret __stdcall
#else
#define EXPORT_API(ret) extern "C" __attribute__((visibility("default"))) ret
#endif

//ignore all such warnings since our stl class will not used internally in the class as private member
#pragma warning(disable : 4251)
class CTimer;
namespace lda {

    class LDADataBlock;
    class LDAModelBlock;
    class SimpleBarrier;
    struct LDAEngineAtomics;
    class LightDocSampler;
    class CBlockedIntQueue;

    // Engine takes care of the entire pipeline of LDA, from reading data to
    // spawning threads, to recording execution time and loglikelihood.
    class LdaEngine {
    public:
        LdaEngine();
        LdaEngine(int numTopic,
            int numVocab,
            float alphaSum,
            float beta,
            int numIter,
            int likelihoodInterval,
            int numThread,
            int mhstep,
            int maxDocToken);

        LdaEngine(int32_t K, int32_t V, int32_t num_threads, int32_t compute_ll_interval, float beta, int32_t num_iterations, int32_t mh_step, float alpha_sum, int maxDocToken);

        ~LdaEngine();


        void InitializeBeforeTest();
        bool InitializeBeforeTrain();
        void AllocateDataMemory(int num_document, int64_t corpus_size);
        void AllocateModelMemory(const LDADataBlock* data_block); //in this case, model memory is allocated according to the datablock;
        void AllocateModelMemory(int num_vocabs, int num_topics, int64_t nonzero_num);
        void AllocateModelMemory(int num_vocabs, int num_topics, int64_t mem_block_size, int64_t alias_mem_block_size);
        //void SetNonzeroNum(int32_t word_id, int32_t nonzero_num);
        //void FinallizeNonzero();
        void SetAlphaSum(float avgDocLength); //alphasum parameter is set by avgdoclength * alpha

        //IO, data
        bool ClearData();  //for clean up training data
        bool ClearModel(); //for testing purpose, before calling SetWordTopic, please clear the old model

        int FeedInData(int* term_id, int* term_freq, int32_t term_num, int32_t vocab_size);
        int FeedInDataDense(int* term_freq, int32_t term_num, int32_t vocab_size);

        //IO, model 
        // NOTE: assume pTopic and pProb are allocated outside the function
        // the length returned will be capped by the pass-in initial value of length(usually it's the size of preallocated memory for pTopic&pProb
        void GetWordTopic(int32_t wordId, int32_t* pTopic, int32_t* pProb, int32_t& length);
        void SetWordTopic(int32_t wordId, int32_t* pTopic, int32_t* pProb, int32_t length);
        void GetModelStat(int64_t &memBlockSize, int64_t &aliasMemBlockSize);
        void GetTopicSummary(int32_t topicId, int32_t* pWords, float* pProb, int32_t& length);

        //mutlithread train/test with the data inside the engine
        void Train(const char* pTrainOutput = nullptr);
        void Test(int32_t burnin_iter, float* pLoglikelihood);

        //testing on single doc
        void TestOneDoc(int* term_id, int* term_freq, int32_t term_num, int* pTopics, int* pProbs, int32_t& numTopicsMax, int32_t numBurnIter, bool reset);
        void TestOneDocDense(int* term_freq, int32_t term_num, int* pTopics, int* pProbs, int32_t& numTopicsMax, int32_t numBurnIter, bool reset);
        void GetDocTopic(int docID, int* pTopic, int* pProb, int32_t& numTopicReturn); // use this function to get the doc's topic output in batch testing scenario

        //output model(word topic) and doc topic
        void DumpFullModel(const std::string& word_topic_dump);
        void DumpDocTopicTable(const std::string& doc_topic_file);

    private:
        double EvalLogLikelihood(bool is_train, int thread_id, int iter, LightDocSampler &sampler);

    private:  // private data
        void Training_Thread();
        void Testing_Thread();
        void CheckFunction(int thread_id, CTimer& tmDebug, const char* msg, bool waitBarrier = true);

        // Number of topics
        int32_t K_;
        // Number of vocabs.
        int32_t V_;

        int32_t compute_ll_interval_;
        int32_t num_threads_;
        int32_t num_iterations_;
        int32_t burnin_iterations_;
        int32_t mh_step_;
        float beta_;
        float alpha_sum_;
        float beta_sum_;
        int maxDocToken_;
        bool bAlphaSumMultiplied; //used to check whether alpha_sum_ is real alpha sum but not alpha
        std::vector<int32_t> word_range_for_each_thread_;

        LDAEngineAtomics* atomic_stats_;
        SimpleBarrier* process_barrier_;         // Local barrier across threads.

        LDADataBlock* data_block_;
        LDAModelBlock* model_block_;

        std::vector<lda::hybrid_map> global_word_topic_table_;
        std::vector<lda::hybrid_alias_map> global_alias_k_v_;
        std::vector<int64_t> global_summary_row_;

        // for generating alias table of beta term
        wood::AliasMultinomialRNGInt alias_rng_int_;
        int32_t beta_height_;
        float beta_mass_;
        std::vector<wood::alias_k_v> beta_k_v_;

        LightDocSampler **samplers_;
        float* likelihood_in_iter_;

        // For TestDocSafe purpose
        int32_t **document_buffer_;

        wood::xorshift_rng rng_;
        CBlockedIntQueue *samplerQueue_;
    };
}   // namespace lda

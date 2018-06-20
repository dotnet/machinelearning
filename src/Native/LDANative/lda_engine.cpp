#include <unordered_map>
#include <cstdint>
#include <string>
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <mutex>
#include <set>
#include <fstream>
#include <thread>
#include <algorithm>
#include <stdlib.h>

#include "timer.h"
#include "rand_int_rng.h"
#include "lda_document.h"
#include "data_block.h"
#include "model_block.h"
#include "lda_engine.hpp"
#include "utils.hpp"
#include "simple_barrier.h"
#include "light_doc_sampler.hpp"

#ifdef _MSC_VER
#include "windows.h"
#else
#include "sched.h"
#endif

namespace lda {
    LdaEngine::LdaEngine(int numTopic,
        int numVocab,
        float alphaSum,
        float beta,
        int numIter,
        int likelihoodInterval,
        int numThread,
        int mhstep,
        int maxDocToken)
        : K_(numTopic),
        V_(numVocab),
        compute_ll_interval_(likelihoodInterval),
        beta_(beta),
        num_iterations_(numIter),
        mh_step_(mhstep),
        alpha_sum_(alphaSum),
        maxDocToken_(maxDocToken),
        samplers_(nullptr),
        document_buffer_(nullptr)
    {
        if (numThread > 0)
        {
            num_threads_ = numThread;
        }
        else
        {
            unsigned int uNumCPU = std::thread::hardware_concurrency();
            num_threads_ = std::max(1, (int)(uNumCPU - 2));
        }
        printf("using %d thread(s) to do train/test\n", num_threads_);

        bAlphaSumMultiplied = false;
        atomic_stats_ = new LDAEngineAtomics();
        model_block_ = new LDAModelBlock();
        data_block_ = new LDADataBlock(num_threads_);
        process_barrier_ = new SimpleBarrier(num_threads_);
        samplerQueue_ = new CBlockedIntQueue();

        document_buffer_ = new int32_t*[num_threads_];
        for (int i = 0; i < num_threads_; i++)
            document_buffer_[i] = new int32_t[maxDocToken_ * 2 + 1];

        likelihood_in_iter_ = nullptr;

        beta_sum_ = beta_ * V_;
    }

    LdaEngine::LdaEngine(int32_t K, int32_t V, int32_t num_threads, int32_t compute_ll_interval, float beta, int32_t num_iterations, int32_t mh_step, float alpha_sum, int maxDocToken)
        : K_(K),
        V_(V),
        compute_ll_interval_(compute_ll_interval),
        beta_(beta),
        num_iterations_(num_iterations),
        mh_step_(mh_step),
        alpha_sum_(alpha_sum),
        maxDocToken_(maxDocToken),
        samplers_(nullptr),
        document_buffer_(nullptr)
    {
        if (num_threads > 0)
        {
            num_threads_ = num_threads;
        }
        else
        {
            unsigned int uNumCPU = std::thread::hardware_concurrency();
            num_threads_ = std::max(1, (int)(uNumCPU - 2));
        }
        bAlphaSumMultiplied = false;
        process_barrier_ = new SimpleBarrier(num_threads_);
        atomic_stats_ = new LDAEngineAtomics();
        data_block_ = new LDADataBlock(num_threads_);
        model_block_ = new LDAModelBlock();
        samplerQueue_ = new CBlockedIntQueue();

        document_buffer_ = new int32_t*[num_threads_];
        for (int i = 0; i < num_threads_; i++)
            document_buffer_[i] = new int32_t[maxDocToken_ * 2 + 1];

        likelihood_in_iter_ = nullptr;
        beta_sum_ = beta_ * V_;

        //set up some trainig parameter, e.g. topicNum, docNum, 
        //model_block_.Read(meta_name); 		
    }


    LdaEngine::~LdaEngine()
    {
        //delete memory space
        delete process_barrier_;
        process_barrier_ = nullptr;

        delete data_block_;
        data_block_ = nullptr;

        delete atomic_stats_;
        atomic_stats_ = nullptr;

        delete model_block_;
        model_block_ = nullptr;

        delete samplerQueue_;
        samplerQueue_ = nullptr;

        for (int i = 0; i < num_threads_; ++i)
        {
            delete samplers_[i];
        }
        delete[] samplers_;

        if (document_buffer_)
        {
            for (int i = 0; i < num_threads_; ++i)
            {
                delete[]document_buffer_[i];
                document_buffer_[i] = nullptr;
            }
            delete[]document_buffer_;
            document_buffer_ = nullptr;
        }

        if (likelihood_in_iter_)
        {
            delete[] likelihood_in_iter_;
            likelihood_in_iter_ = nullptr;
        }
    }

    bool LdaEngine::InitializeBeforeTrain()
    {
        CTimer tmDebug(true);
        CheckFunction(0, tmDebug, "enter initializeBeforeTrain", false);
        //allocate model memory from the data preloaded
        AllocateModelMemory(data_block_);
        CheckFunction(0, tmDebug, "allocate model memory", false);

        double alloc_start = lda::get_time();
        global_word_topic_table_.resize(V_);
        alias_rng_int_.Init(K_);
        beta_k_v_.resize(K_);
        global_alias_k_v_.resize(V_);

        for (int i = 0; i < V_; ++i)
        {
            global_alias_k_v_[i] = model_block_->get_alias_row(i);
        }
        global_summary_row_.resize(K_);
        CheckFunction(0, tmDebug, "initlaizing global tables used in sampling", false);

        word_range_for_each_thread_.resize(num_threads_ + 1);
        int32_t word_num_each_thread = V_ / num_threads_;
        word_range_for_each_thread_[0] = 0;
        for (int32_t i = 0; i < num_threads_ - 1; ++i)
        {
            word_range_for_each_thread_[i + 1] = word_range_for_each_thread_[i] + word_num_each_thread;
        }
        word_range_for_each_thread_[num_threads_] = V_;

        //setup sampler
        samplers_ = new LightDocSampler*[num_threads_];
        samplerQueue_->clear();

        for (int i = 0; i < num_threads_; ++i)
        {
            samplers_[i] = new LightDocSampler(
                K_,
                V_,
                num_threads_,
                mh_step_,
                beta_,
                alpha_sum_,
                global_word_topic_table_,
                global_summary_row_,
                global_alias_k_v_,
                beta_height_,
                beta_mass_,
                beta_k_v_);

            samplerQueue_->push(i);
        }
        CheckFunction(0, tmDebug, "create samplers", false);
        return true;
    }

    void LdaEngine::InitializeBeforeTest()
    {
        // TODO(jiyuan):
        // 1, Allocating space for word-topic-table and alias table according to the input data of SetModel interface (done)
        // 2, Create multiple thread-specific sampler
        // 3, set word_range_for_each_thread_
        // Adjust the alpha_sum_ parameter for each thread-specific sampler
        CTimer tmDebug(true);
        CheckFunction(0, tmDebug, "enter initializeBeforeTest", false);

        global_word_topic_table_.resize(V_);
        alias_rng_int_.Init(K_);
        beta_k_v_.resize(K_);
        global_alias_k_v_.resize(V_);

        for (int i = 0; i < V_; ++i)
        {
            global_alias_k_v_[i] = model_block_->get_alias_row(i);
        }
        CheckFunction(0, tmDebug, "initlaizing global tables used in sampling", false);

        // Set the word range for each thread
        word_range_for_each_thread_.resize(num_threads_ + 1);
        int32_t word_num_each_thread = V_ / num_threads_;
        word_range_for_each_thread_[0] = 0;
        for (int32_t i = 0; i < num_threads_ - 1; ++i)
        {
            word_range_for_each_thread_[i + 1] = word_range_for_each_thread_[i] + word_num_each_thread;
        }
        word_range_for_each_thread_[num_threads_] = V_;

        //setup sampler
        if (samplers_)
        {
            for (int i = 0; i < num_threads_; ++i)
            {
                delete samplers_[i];
            }
            delete[] samplers_;
        }
        if (document_buffer_)
        {
            for (int i = 0; i < num_threads_; ++i)
            {
                delete[]document_buffer_[i];
                document_buffer_[i] = nullptr;
            }
            delete[]document_buffer_;
            document_buffer_ = nullptr;
        }

        samplers_ = new LightDocSampler*[num_threads_];
        document_buffer_ = new int32_t*[num_threads_];
        samplerQueue_->clear();

        for (int i = 0; i < num_threads_; ++i)
        {
            samplers_[i] = new LightDocSampler(
                K_,
                V_,
                num_threads_,
                mh_step_,
                beta_,
                alpha_sum_,
                global_word_topic_table_,
                global_summary_row_,
                global_alias_k_v_,
                beta_height_,
                beta_mass_,
                beta_k_v_);

            samplers_[i]->AdaptAlphaSum(false);
            document_buffer_[i] = new int32_t[maxDocToken_ * 2 + 1];

            samplerQueue_->push(i);
        }
        CheckFunction(0, tmDebug, "create samplers", false);

        // build alias table
        // 1, build alias table for the dense term,  beta_k_v_, which is shared by all the words
        beta_mass_ = 0;
        std::vector<float> proportion(K_);
        for (int k = 0; k < K_; ++k)
        {
            proportion[k] = beta_ / (global_summary_row_[k] + beta_sum_);
            beta_mass_ += proportion[k];
        }
        alias_rng_int_.SetProportionMass(proportion, beta_mass_, beta_k_v_, &beta_height_, samplers_[0]->rng());

        // 2,  build alias table for the sparse term
        for (int thread_id = 0; thread_id < num_threads_; ++thread_id)
        {
            LightDocSampler &sampler = *(samplers_[thread_id]);
            sampler.build_alias_table(word_range_for_each_thread_[thread_id], word_range_for_each_thread_[thread_id + 1], thread_id);
        }
        CheckFunction(0, tmDebug, "build alisa table", false);
    }

    void LdaEngine::Train(const char* pTrainOutput)
    {
        std::vector<std::thread> threads(num_threads_);
        atomic_stats_->thread_counter_ = 0;

        for (auto& thr : threads) {
            thr = std::thread(&LdaEngine::Training_Thread, this);
        }

        printf("started training with %d threads\n", num_threads_);
        for (auto& thr : threads) {
            thr.join();
        }

        if (pTrainOutput)
        {
            DumpDocTopicTable(pTrainOutput);
        }
    }

    void LdaEngine::Test(int32_t burnin_iter, float* pLoglikelihood)
    {
        std::vector<std::thread> threads(num_threads_);
        atomic_stats_->thread_counter_ = 0;
        burnin_iterations_ = burnin_iter;

        likelihood_in_iter_ = new float[burnin_iterations_];
        for (int i = 0; i < burnin_iterations_; i++)
        {
            likelihood_in_iter_[i] = 0.0;
        }

        for (auto& thr : threads) {
            thr = std::thread(&LdaEngine::Testing_Thread, this);
        }

        printf("started testing with %d threads\n", num_threads_);

        for (auto& thr : threads) {
            thr.join();
        }

        //get the loglikelihood of each burn in iteration
        for (int i = 0; i < burnin_iterations_; i++)
        {
            pLoglikelihood[i] = likelihood_in_iter_[i]; //just set an arbitary value here for later update
        }
    }

    void LdaEngine::CheckFunction(int thread_id, CTimer &tmDebug, const char* msg, bool waitBarrier)
    {
        /*if (thread_id == 0)
        {
        sprintf(tmDebug.m_szMessage, msg);
        tmDebug.InnerTag();
        system("pause");
        }

        if (waitBarrier)
        {
        process_barrier_->wait();
        }*/
    }

    void LdaEngine::Training_Thread()
    {
        CTimer tmDebug(true);

        int thread_id = atomic_stats_->thread_counter_++;
        std::vector<std::pair<int, double>> llcontainer;
        // Set core affinity which helps performance improvement
#ifdef _MSC_VER
        long long maskLL = 0;
        maskLL |= (1LL << (thread_id));
        DWORD_PTR mask = maskLL;
        SetThreadAffinityMask(GetCurrentThread(), mask);
#elif !defined(__APPLE__)
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(thread_id, &set);
        sched_setaffinity(0, sizeof(cpu_set_t), &set);
#endif
        // Each thread builds a portion of word-topic table. We do this way because each word-topic row 
        // has a thread-specific buffer for rehashing
        process_barrier_->wait();
        LightDocSampler &sampler_ = *(samplers_[thread_id]);
        sampler_.AdaptAlphaSum(true);

        sampler_.build_word_topic_table(thread_id, num_threads_, *model_block_);
        process_barrier_->wait();
        CheckFunction(thread_id, tmDebug, "intialize word_topic_table for sampler - in function train_thread");

        int32_t token_num = 0;
        int32_t doc_start = data_block_->Begin(thread_id);
        int32_t doc_end = data_block_->End(thread_id);

        for (int32_t doc_index = doc_start; doc_index != doc_end; ++doc_index)
        {
            std::shared_ptr<LDADocument> doc = data_block_->GetOneDoc(doc_index);
            int doc_size = doc->size();
            for (int i = 0; i < doc_size; ++i)
            {
                int topic = sampler_.rand_k();
                doc->SetTopic(i, topic);
            }
            int cursor = doc->get_cursor();
            token_num += sampler_.GlobalInit(doc.get());
        }
        process_barrier_->wait();
        CheckFunction(thread_id, tmDebug, "intialize token topic before iterations - in function train_thread");

        for (int i = 0; i < num_threads_; ++i)
        {
            std::vector<word_topic_delta>& wtd_vec = samplers_[i]->get_word_topic_delta(thread_id);
            for (auto& wtd : wtd_vec)
            {
                global_word_topic_table_[wtd.word].inc(wtd.topic, wtd.delta);
            }
        }
        process_barrier_->wait();
        CheckFunction(thread_id, tmDebug, "intialize word topic model before iterations - in function train_thread");

        // use thread-private delta table to get global table
        {
            std::lock_guard<std::mutex> lock(atomic_stats_->global_mutex_);

            std::vector<int64_t> &summary = sampler_.get_delta_summary_row();
            for (int i = 0; i < K_; ++i)
            {
                global_summary_row_[i] += summary[i];
            }
        }
        process_barrier_->wait();
        CheckFunction(thread_id, tmDebug, "global summary & Complete setup train before iterations - in function train_thread");

        for (int iter = 0; iter < num_iterations_; ++iter)
        {
            CheckFunction(thread_id, tmDebug, "----------------------iteration start - in function train_thread---------------------");
            int32_t token_sweeped = 0;
            atomic_stats_->num_tokens_clock_ = 0;
            // build alias table
            // 1, build alias table for the dense term,  beta_k_v_, which is shared by all the words
            if (thread_id == 0)
            {
                beta_mass_ = 0;
                std::vector<float> proportion(K_);
                for (int k = 0; k < K_; ++k)
                {
                    proportion[k] = beta_ / (global_summary_row_[k] + beta_sum_);
                    beta_mass_ += proportion[k];
                }

                alias_rng_int_.SetProportionMass(proportion, beta_mass_, beta_k_v_, &beta_height_, sampler_.rng());
            }
            process_barrier_->wait();
            CheckFunction(thread_id, tmDebug, "built alias table dense - in function train_thread");

            // 2,  build alias table for the sparse term
            sampler_.build_alias_table(word_range_for_each_thread_[thread_id], word_range_for_each_thread_[thread_id + 1], thread_id);
            process_barrier_->wait();
            CheckFunction(thread_id, tmDebug, "built alias table sparse - in function train_thread");

            sampler_.EpocInit();
            process_barrier_->wait();
            CheckFunction(thread_id, tmDebug, "EpochInit - in function train_thread");

            //3. main part of the training - sampling over documents in this iteration
            double iter_start = lda::get_time();
            int32_t doc_start_local = data_block_->Begin(thread_id);
            int32_t doc_end_local = data_block_->End(thread_id);

            for (int32_t doc_index = doc_start_local; doc_index != doc_end_local; ++doc_index)
            {
                std::shared_ptr<LDADocument> doc = data_block_->GetOneDoc(doc_index);
                token_sweeped += sampler_.SampleOneDoc(doc.get());
            }
            atomic_stats_->num_tokens_clock_ += token_sweeped;

            process_barrier_->wait();
            double iter_end = lda::get_time();

            if (thread_id == 0)
            {
                double seconds_this_iter = iter_end - iter_start;
                //std::cout << "end sampling, thread = " << thread_id << ", elpased time = " << (seconds_this_iter) << std::endl;

                printf("Iter: %04d", iter);
                std::cout
                    << "\tThread = " << thread_id
                    << "\tTokens: " << atomic_stats_->num_tokens_clock_
                    << "\tTook: " << seconds_this_iter << " sec"
                    << "\tThroughput: "
                    << static_cast<double>(atomic_stats_->num_tokens_clock_) / (seconds_this_iter) << " token/(thread*sec)"
                    << std::endl;
            }
            process_barrier_->wait();
            CheckFunction(thread_id, tmDebug, "train(gibbs sampling) - in function train_thread");

            //4. syncup global table
            double sync_start = lda::get_time();
            for (int i = 0; i < num_threads_; ++i)
            {
                std::vector<word_topic_delta> & wtd_vec = samplers_[i]->get_word_topic_delta(thread_id);
                for (auto& wtd : wtd_vec)
                {
                    global_word_topic_table_[wtd.word].inc(wtd.topic, wtd.delta);
                }
            }

            // use thread-private delta table to update global table
            {
                std::lock_guard<std::mutex> lock(atomic_stats_->global_mutex_);
                std::vector<int64_t> &summary = sampler_.get_delta_summary_row();
                for (int i = 0; i < K_; ++i)
                {
                    global_summary_row_[i] += summary[i];
                }
            }
            process_barrier_->wait();
            CheckFunction(thread_id, tmDebug, "syncup global word_topic table - in function train_thread");

            if (compute_ll_interval_ != -1 && (iter % compute_ll_interval_ == 0 || iter == num_iterations_ - 1))
            {
                double ll = EvalLogLikelihood(true, thread_id, iter, sampler_);
                llcontainer.push_back(std::pair<int, double>(iter, ll));
            }

            CheckFunction(thread_id, tmDebug, "----------------------iteration end - in function train_thread---------------------");
        }

        if (thread_id == 0)
        {
            //output the ll once
            for (int i = 0; i < llcontainer.size(); i++)
            {
                printf("loglikelihood @iter%04d = %f\n", llcontainer[i].first, llcontainer[i].second);
            }
        }

        process_barrier_->wait();

        snprintf(tmDebug.m_szMessage, 200, "thread_id = %d, training iterations", thread_id);
        tmDebug.InnerTag();
    }

    void LdaEngine::Testing_Thread()
    {
        int thread_id = atomic_stats_->thread_counter_++;

        // Set core affinity which helps performance improvement
#ifdef _MSC_VER
        long long maskLL = 0;
        maskLL |= (1LL << (thread_id));
        DWORD_PTR mask = maskLL;
        SetThreadAffinityMask(GetCurrentThread(), mask);
#elif !defined(__APPLE__)
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(thread_id, &set);
        sched_setaffinity(0, sizeof(cpu_set_t), &set);
#endif
        process_barrier_->wait();

        //// Each thread builds a portion of word-topic table. We do this way because each word-topic row 
        //// has a thread-specific buffer for rehashing
        LightDocSampler &sampler_ = *(samplers_[thread_id]);
        sampler_.AdaptAlphaSum(false);

        //sampler_.build_word_topic_table(thread_id, num_threads_, model_block_);
        //process_barrier_->wait();

        double init_start = lda::get_time();
        int32_t token_num = 0;
        int32_t doc_start = data_block_->Begin(thread_id);
        int32_t doc_end = data_block_->End(thread_id);

        for (int32_t doc_index = doc_start; doc_index != doc_end; ++doc_index)
        {
            std::shared_ptr<LDADocument> doc = data_block_->GetOneDoc(doc_index);
            int doc_size = doc->size();
            for (int i = 0; i < doc_size; ++i)
            {
                int topic = sampler_.rand_k();
                doc->SetTopic(i, topic);
            }
            int cursor = doc->get_cursor();
            token_num += sampler_.GlobalInit(doc.get());
        }

        /*double init_end = lda::get_time();
        printf("Thread ID = %d, token num = %d, Init took %fsec, Throughput: %f(token/sec)\n",
        thread_id,
        token_num,
        init_end - init_start,
        static_cast<double>(token_num) / (init_end - init_start));*/

        process_barrier_->wait();

        /*if (thread_id == 0)
        {
        printf("Global Init OK");
        printf("Start aggreating word_topic_delta, thread = %d\n", thread_id);
        }
        process_barrier_->wait();*/

        // build alias table
        // 1, build alias table for the dense term,  beta_k_v_, which is shared by all the words
        if (thread_id == 0)
        {
            beta_mass_ = 0;
            std::vector<float> proportion(K_);
            for (int k = 0; k < K_; ++k)
            {
                proportion[k] = beta_ / (global_summary_row_[k] + beta_sum_);
                beta_mass_ += proportion[k];
            }

            alias_rng_int_.SetProportionMass(proportion, beta_mass_, beta_k_v_, &beta_height_, sampler_.rng());
            //std::cout << "Start build alias table" << std::endl;
        }

        // 2,  build alias table for the sparse term
        double alias_start = lda::get_time();
        process_barrier_->wait();
        sampler_.build_alias_table(word_range_for_each_thread_[thread_id], word_range_for_each_thread_[thread_id + 1], thread_id);
        process_barrier_->wait();

        /*double alias_end = lda::get_time();
        if (thread_id == 0)
        {
        std::cout << "Elapsed time for building alias table: " << (alias_end - alias_start) << std::endl;
        }
        process_barrier_->wait();*/

        // print the log-likelihood before inference
        EvalLogLikelihood(true, thread_id, 0, sampler_);

        double total_start = lda::get_time();
        for (int iter = 0; iter < burnin_iterations_; ++iter)
        {
            double iter_start = lda::get_time();
            int32_t token_sweeped = 0;
            atomic_stats_->num_tokens_clock_ = 0;
            int32_t doc_start_local = data_block_->Begin(thread_id);
            int32_t doc_end_local = data_block_->End(thread_id);

            for (int32_t doc_index = doc_start_local; doc_index != doc_end_local; ++doc_index)
            {
                std::shared_ptr<LDADocument> doc = data_block_->GetOneDoc(doc_index);
                token_sweeped += sampler_.InferOneDoc(doc.get());
            }
            atomic_stats_->num_tokens_clock_ += token_sweeped;

            process_barrier_->wait();
            double iter_end = lda::get_time();

            if (thread_id == 0)
            {
                double seconds_this_iter = iter_end - iter_start;

                printf("Iter: %04d", iter);
                std::cout
                    << "\tThread = " << thread_id
                    << "\tTokens: " << atomic_stats_->num_tokens_clock_
                    << "\tTook: " << seconds_this_iter << " sec"
                    << "\tThroughput: "
                    << static_cast<double>(atomic_stats_->num_tokens_clock_) / (seconds_this_iter) << " token/(thread*sec)"
                    << std::endl;

            }

            process_barrier_->wait();

            if (compute_ll_interval_ != -1 && (iter % compute_ll_interval_ == 0 || iter == burnin_iterations_ - 1))
            {
                EvalLogLikelihood(false, thread_id, iter, sampler_);
            }
        }

        double total_end = lda::get_time();
        printf("thread_id = %d, Total time for burnin iterations : %f sec.\n", thread_id, total_end - total_start);
    }

    void LdaEngine::AllocateDataMemory(int num_document, int64_t corpus_size)
    {
        data_block_->Allocate(num_document, corpus_size);
    }

    void LdaEngine::AllocateModelMemory(const LDADataBlock* data_block)
    {
        model_block_->InitFromDataBlock(data_block, V_, K_);

        global_word_topic_table_.resize(V_);

        for (int i = 0; i < V_; ++i)
        {
            global_word_topic_table_[i] = model_block_->get_row(i, nullptr);
        }
    }

    void LdaEngine::AllocateModelMemory(int num_vocabs, int num_topics, int64_t nonzero_num)
    {
        model_block_->Init(num_vocabs, num_topics, nonzero_num);

        global_word_topic_table_.resize(num_vocabs);

        for (int i = 0; i < num_vocabs; ++i)
        {
            global_word_topic_table_[i] = model_block_->get_row(i, nullptr);
        }
    }

    void LdaEngine::AllocateModelMemory(int num_vocabs, int num_topics, int64_t mem_block_size, int64_t alias_mem_block_size)
    {
        model_block_->Init(num_vocabs, num_topics, mem_block_size, alias_mem_block_size); //memory allocated here

        global_word_topic_table_.resize(num_vocabs);
        global_summary_row_.resize(K_, 0);

        //each value inside the global_word_topic_table_ will be set while call SetWordTopic()
    }

    int LdaEngine::FeedInData(int* term_id, int* term_freq, int32_t term_num, int32_t vocab_size)
    {
        // bool b_ret = true;
        if (V_ == 0) //number vocab could be set in allocating model memory function
            V_ = vocab_size;

        //data_block represent for one doc
        return data_block_->Add(term_id, term_freq, term_num);
        // return b_ret;
    }

    int LdaEngine::FeedInDataDense(int* term_freq, int32_t term_num, int32_t vocab_size)
    {
        // bool b_ret = true;
        if (V_ == 0) //number vocab could be set in allocating model memory function
            V_ = vocab_size;

        //data_block represent for one doc
        return data_block_->AddDense(term_freq, term_num);
        // return b_ret;
    }

    void LdaEngine::TestOneDoc(int* term_id, int* term_freq, int32_t term_num, int* pTopics, int* pProbs, int32_t& numTopicsMax, int32_t numBurnIter, bool reset)
    {
        //numTopicsMax initialy holds the max returned topic number in order to hold the pTopic/pProbs memory in outside function
        //when data return, numTopicsMax should contains the real topic number returned.
        int sampler_id = 0;
        sampler_id = samplerQueue_->pop();

        LightDocSampler &sampler = *(samplers_[sampler_id]);
        int64_t data_length = 1;
        for (int i = 0; i < term_num; ++i)
        {
            for (int j = 0; j < term_freq[i]; ++j)
            {
                data_length += 2;
            }
        }

        assert(data_length <= maxDocToken_ * 2 + 1);

        if (reset)
        {
            // restart the rng seeds, so that we always get consistent result for the same input
            rng_.restart();
            sampler.rng_restart();
        }

        // NOTE(jiyuan): in multi-threaded implementation, the dynamic memory allocation
        // may cause contention at OS heap lock
        // int32_t *document_buffer = new int32_t[data_length];
        int64_t idx = 1;
        for (int i = 0; i < term_num; ++i)
        {
            for (int j = 0; j < term_freq[i]; ++j)
            {
                document_buffer_[sampler_id][idx++] = term_id[i];
                document_buffer_[sampler_id][idx++] = rng_.rand_k(K_);
            }
        }

        std::shared_ptr<LDADocument> doc(new LDADocument(document_buffer_[sampler_id], document_buffer_[sampler_id] + data_length));

        for (int iter = 0; iter < numBurnIter; ++iter)
        {
            sampler.InferOneDoc(doc.get());
        }
        sampler.GetDocTopic(doc.get(), pTopics, pProbs, numTopicsMax);

        samplerQueue_->push(sampler_id);
    }

    void LdaEngine::TestOneDocDense(int* term_freq, int32_t term_num, int* pTopics, int* pProbs, int32_t& numTopicsMax, int32_t numBurnIter, bool reset)
    {
        //numTopicsMax initialy holds the max returned topic number in order to hold the pTopic/pProbs memory in outside function
        //when data return, numTopicsMax should contains the real topic number returned.
        int sampler_id = 0;
        sampler_id = samplerQueue_->pop();

        LightDocSampler &sampler = *(samplers_[sampler_id]);
        int64_t data_length = 1;
        for (int i = 0; i < term_num; ++i)
        {
            for (int j = 0; j < term_freq[i]; ++j)
            {
                data_length += 2;
            }
        }

        assert(data_length <= maxDocToken_ * 2 + 1);

        if (reset)
        {
            // restart the rng seeds, so that we always get consistent result for the same input
            rng_.restart();
            sampler.rng_restart();
        }

        // NOTE(jiyuan): in multi-threaded implementation, the dynamic memory allocation
        // may cause contention at OS heap lock
        // int32_t *document_buffer = new int32_t[data_length];
        int64_t idx = 1;
        for (int i = 0; i < term_num; ++i)
        {
            for (int j = 0; j < term_freq[i]; ++j)
            {
                document_buffer_[sampler_id][idx++] = i;
                document_buffer_[sampler_id][idx++] = rng_.rand_k(K_);
            }
        }

        std::shared_ptr<LDADocument> doc(new LDADocument(document_buffer_[sampler_id], document_buffer_[sampler_id] + data_length));

        for (int iter = 0; iter < numBurnIter; ++iter)
        {
            sampler.InferOneDoc(doc.get());
        }
        sampler.GetDocTopic(doc.get(), pTopics, pProbs, numTopicsMax);

        samplerQueue_->push(sampler_id);
    }

    void LdaEngine::GetDocTopic(int docID, int* pTopic, int* pProb, int32_t& numTopicReturn)
    {
        //to be added by jinhui
        //get the current topic vector of the document
        int thread_id = 0;
        LightDocSampler &sampler = *(samplers_[thread_id]);

        sampler.GetDocTopic(data_block_->GetOneDoc(docID).get(), pTopic, pProb, numTopicReturn);
    }

    void LdaEngine::SetAlphaSum(float avgDocLength)
    {
        if (!bAlphaSumMultiplied)
        {
            alpha_sum_ = alpha_sum_ * avgDocLength;
            bAlphaSumMultiplied = true;
        }
        printf("alpha_sum was set to %f", alpha_sum_);
    }

    bool LdaEngine::ClearData()
    {
        data_block_->Clear();
        return true;
    }

    bool LdaEngine::ClearModel()
    {
        model_block_->Clear();
        return true;
    }

    //function to support dumping the topic_model model file
    void LdaEngine::GetWordTopic(int32_t wordId, int32_t* pTopic, int32_t* pProb, int32_t& length)
    {
        //cap the topic number here according to inpassed value of length
        int lengthCap = length;

        // NOTE(jiyuan): we MUST check whether the word-topic row is empty before get its value
        if (global_word_topic_table_[wordId].capacity() == 0)
        {
            length = 0;
            return;
        }

        length = 0;
        for (int i = 0; i < K_; ++i)
        {
            if (global_word_topic_table_[wordId][i] > 0)
            {
                pTopic[length] = i;
                pProb[length] = global_word_topic_table_[wordId][i];
                length++;

                if (length >= lengthCap)
                    break;
            }
        }
    }

    // Compare by frequencies in descending order.
    bool CompareTerms(const std::pair<int, int> &term1, const std::pair<int, int> &term2)
    {
        // REVIEW wenhanw(yaeld): consider changing this to impose a total order, since quicksort is not stable.
        return term2.second < term1.second;
    }

    void LdaEngine::GetTopicSummary(int32_t topicId, int32_t* pWords, float* pProb, int32_t& length)
    {
        std::vector<std::pair<int, int>> allTermsVec;
        int sumFreq = 0;
        for (int i = 0; i < V_; i++) //for all the terms check the topic distribution
        {
            if (global_word_topic_table_[i][topicId] > 0)
            {
                std::pair<int, int> p;
                p.first = i;
                p.second = global_word_topic_table_[i][topicId];
                allTermsVec.push_back(p);
                sumFreq += global_word_topic_table_[i][topicId];
            }
        }

        std::sort(allTermsVec.begin(), allTermsVec.end(), CompareTerms);

        int usedTerm = (int)allTermsVec.size();
        length = std::min(usedTerm, length);
        for (int i = 0; i < length; i++)
        {
            pWords[i] = allTermsVec[i].first;
            pProb[i] = (((float)(allTermsVec[i].second)) + beta_) / (sumFreq + beta_ * V_);
        }
    }

    //function to support loading the topic_model model file
    void LdaEngine::SetWordTopic(int32_t wordId, int32_t* pTopic, int32_t* pProb, int32_t length)
    {
        //Note: taifengw(jinhui) whether we should really use the "true" here
        model_block_->SetWordInfo(wordId, length, true);
        global_word_topic_table_[wordId] = model_block_->get_row(wordId, nullptr);

        for (int i = 0; i < length; ++i)
        {
            global_word_topic_table_[wordId].inc(pTopic[i], pProb[i]);
            global_summary_row_[pTopic[i]] += pProb[i];
        }
    }

    void LdaEngine::GetModelStat(int64_t &memBlockSize, int64_t &aliasMemBlockSize)
    {
        //Note: taifengw, get the model's value at the end of training stage. try to save these two numbers to disk file
        model_block_->GetModelStat(memBlockSize, aliasMemBlockSize);
    }

    double LdaEngine::EvalLogLikelihood(bool is_train, int thread_id, int iter, LightDocSampler &sampler)
    {
        double doc_ll = 0;
        double word_ll = 0;

        if (thread_id == 0)
        {
            atomic_stats_->doc_ll_ = 0;
            atomic_stats_->word_ll_ = 0;
        }
        process_barrier_->wait();

        int doc_num = 0;
        int32_t doc_start = data_block_->Begin(thread_id);
        int32_t doc_end = data_block_->End(thread_id);
        for (int32_t doc_index = doc_start; doc_index != doc_end; ++doc_index)
        {
            std::shared_ptr<LDADocument> doc = data_block_->GetOneDoc(doc_index);
            doc_ll += sampler.ComputeOneDocLLH(doc.get());
            doc_num++;
        }
        atomic_stats_->doc_ll_ = atomic_stats_->doc_ll_ + doc_ll;
        process_barrier_->wait();

        word_ll = sampler.ComputeWordLLH(word_range_for_each_thread_[thread_id], word_range_for_each_thread_[thread_id + 1]);
        atomic_stats_->word_ll_ = atomic_stats_->word_ll_ + word_ll;
        process_barrier_->wait();

        double total_ll = 0;
        if (thread_id == 0)
        {
            double normalized_ll = sampler.NormalizeWordLLH();

            total_ll = 0;
            total_ll += atomic_stats_->doc_ll_;
            total_ll += atomic_stats_->word_ll_;
            total_ll += normalized_ll;

            if (!is_train)
            {
                likelihood_in_iter_[iter] = (float)total_ll;
            }

            std::cout << "Total likelihood: " << total_ll << "\t";
            std::cout << "..........[Nomralized word ll: " << normalized_ll << "\t"
                << "Word  likelihood: " << atomic_stats_->word_ll_ << "\t"
                << "Doc   likelihood: " << atomic_stats_->doc_ll_ << "]" << std::endl;
        }
        process_barrier_->wait();

        return total_ll;
    }

    void LdaEngine::DumpDocTopicTable(const std::string& doc_topic_file)
    {
        std::ofstream dt_stream;
        dt_stream.open(doc_topic_file, std::ios::out);
        // CHECK(dt_stream.good()) << "Open doc_topic_file fail: " << doc_topic_file;
        assert(dt_stream.good());

        int32_t num_documents = data_block_->num_documents();
        int32_t doc_start = 0;
        int32_t doc_end = num_documents;

        lda::light_hash_map doc_topic_counter_(1024);

        for (int32_t doc_index = doc_start; doc_index != doc_end; ++doc_index)
        {
            std::shared_ptr<LDADocument> doc = data_block_->GetOneDoc(doc_index);
            doc_topic_counter_.clear();
            doc->GetDocTopicCounter(doc_topic_counter_);

            dt_stream << doc_index;
            if (doc->size())
            {
                int32_t capacity = doc_topic_counter_.capacity();
                int32_t *key = doc_topic_counter_.key();
                int32_t *value = doc_topic_counter_.value();
                int32_t nonzero_num = 0;

                for (int i = 0; i < capacity; ++i)
                {
                    if (key[i] > 0)
                    {
                        dt_stream << " " << key[i] - 1 << ":" << value[i];
                    }
                }
            }
            dt_stream << std::endl;
        }
        dt_stream.close();
    }

    void LdaEngine::DumpFullModel(const std::string& word_topic_dump)
    {
        std::ofstream wt_stream;
        wt_stream.open(word_topic_dump, std::ios::out);
        assert(wt_stream.good());

        for (int w = 0; w < V_; ++w)
        {
            int nonzero_num = global_word_topic_table_[w].nonzero_num();
            if (nonzero_num)
            {
                wt_stream << w;
                for (int t = 0; t < K_; ++t)
                {
                    // if (word_topic_table_[w * K_ + t])
                    // if (word_topic_table_[(int64_t)w * K_ + t])
                    if (global_word_topic_table_[w][t] > 0)
                    {
                        wt_stream << " " << t << ":" << global_word_topic_table_[w][t];
                    }
                }
                wt_stream << std::endl;
            }
        }
        wt_stream.close();

        std::ofstream summary_stream;
        summary_stream.open("summary_row.txt", std::ios::out);
        for (int i = 0; i < K_; ++i)
        {
            summary_stream << global_summary_row_[i] << std::endl;
        }
        summary_stream.close();
    }
}   // namespace lda

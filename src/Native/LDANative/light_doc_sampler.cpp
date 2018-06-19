#include <algorithm>
#include <time.h>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>

#include "lda_document.h"
#include "light_doc_sampler.hpp"


namespace lda
{
    LightDocSampler::LightDocSampler(
        int32_t K,
        int32_t V,
        int32_t num_threads,
        int32_t mh_step,
        float beta,
        float alpha_sum,
        std::vector<lda::hybrid_map> &word_topic_table,
        std::vector<int64_t> &summary_row,
        std::vector<lda::hybrid_alias_map> &alias_kv,
        int32_t &beta_height,
        float& beta_mass,
        std::vector<wood::alias_k_v> &beta_k_v)
        : doc_topic_counter_(1024),
        word_topic_table_(word_topic_table), summary_row_(summary_row),
        alias_k_v_(alias_kv),
        beta_height_(beta_height),
        beta_mass_(beta_mass),
        beta_k_v_(beta_k_v),
        K_(K),
        V_(V),
        num_threads_(num_threads),
        mh_step_for_gs_(mh_step),
        beta_(beta),
        alpha_sum_(alpha_sum)
    {
        beta_sum_ = beta_ * V_;
        alpha_ = alpha_sum_ / K_;

        ll_alpha_ = (lda::real_t)0.01;
        ll_alpha_sum_ = ll_alpha_ * K_;

        // Precompute LLH parameters
        log_doc_normalizer_ = LogGamma(ll_alpha_ * K_) - K_ * LogGamma(ll_alpha_);
        log_topic_normalizer_ = LogGamma(beta_sum_) - V_ * LogGamma(beta_);

        alias_rng_.Init(K_);

        q_w_proportion_.resize(K_);
        delta_summary_row_.resize(K_);
        word_topic_delta_.resize(num_threads_);

        rehashing_buf_ = new int32_t[K_ * 2];
    }

    LightDocSampler::~LightDocSampler()
    {
        delete[] rehashing_buf_;
    }

    // Initialize word_topic_table and doc_topic_counter for each doc
    int32_t LightDocSampler::GlobalInit(LDADocument *doc)
    {
        int32_t token_num = 0;
        int32_t doc_size = doc->size();
        for (int i = 0; i < doc_size; ++i)
        {
            int32_t w = doc->Word(i);
            int32_t t = doc->Topic(i);

            word_topic_delta wtd;
            int32_t shard_id = w % num_threads_;
            wtd.word = w;
            wtd.topic = t;
            wtd.delta = 1;
            word_topic_delta_[shard_id].push_back(wtd);

            ++delta_summary_row_[t];

            ++token_num;
        }
        return token_num;
    }

    int32_t LightDocSampler::DocInit(LDADocument *doc)
    {
        int num_words = doc->size();

        // compute the doc_topic_counter on the fly
        doc_topic_counter_.clear();
        doc->GetDocTopicCounter(doc_topic_counter_);

        doc_size_ = num_words;
        n_td_sum_ = (lda::real_t)num_words;

        return 0;
    }

    bool CompareFirstElement(const std::pair<int, int> &p1, const std::pair<int, int> &p2)
    {
        return p1.first < p2.first;
    }

    void LightDocSampler::GetDocTopic(LDADocument *doc, int* pTopics, int* pProbs, int32_t& numTopicsMax)
    {
        doc_topic_counter_.clear();
        doc->GetDocTopicCounter(doc_topic_counter_);

        // note(taifengw), do we have to assume this?
        // probably first sort the topic vector according to the probs and keep the first numTopicsMax topics
        // We assume the numTopicsMax is not less than the length of current document?? or it should be maxiumly the toipc number
        // assert(numTopicsMax >= doc->size());

        int32_t capacity = doc_topic_counter_.capacity();
        int32_t *key = doc_topic_counter_.key();
        int32_t *value = doc_topic_counter_.value();

        std::vector<std::pair<int, int>> vec;
        int32_t idx = 0;
        for (int i = 0; i < capacity; ++i)
        {
            if (key[i] > 0)
            {
                std::pair<int, int> pair;
                pair.first = key[i] - 1;
                pair.second = value[i];
                vec.push_back(pair);
                idx++;

                if (idx == numTopicsMax)
                    break;
            }
        }
        numTopicsMax = idx;
        std::sort(vec.begin(), vec.end(), CompareFirstElement);
        for (int i = 0; i < idx; i++)
        {
            pTopics[i] = vec[i].first;
            pProbs[i] = vec[i].second;
        }
    }

    void LightDocSampler::EpocInit()
    {
        std::fill(delta_summary_row_.begin(), delta_summary_row_.end(), 0);
        for (auto &shard : word_topic_delta_)
        {
            shard.clear();
        }
    }

    void LightDocSampler::AdaptAlphaSum(bool is_train)
    {
        rng_.restart(); //reset the sampler so that we will get deterministic result by different runs, train-test, train-save-test, etc.

        if (is_train)
        {
            if (alpha_sum_ < 10)
            {
                alpha_sum_ = 100;
            }
        }
        else
        {
            if (alpha_sum_ > 10)
            {
                alpha_sum_ = 1;
            }
        }
        alpha_ = alpha_sum_ / K_;
    }

    void LightDocSampler::build_alias_table(int32_t lower, int32_t upper, int thread_id)
    {
        for (int w = lower; w < upper; ++w)
        {
            GenerateAliasTableforWord(w);
        }
    }
    void LightDocSampler::build_word_topic_table(int32_t thread_id, int32_t num_threads, lda::LDAModelBlock &model_block)
    {
        for (int i = 0; i < V_; ++i)
        {
            if (i % num_threads == thread_id)
            {
                word_topic_table_[i] = model_block.get_row(i, rehashing_buf_);
            }
        }
    }

    int32_t LightDocSampler::SampleOneDoc(LDADocument *doc)
    {
        return  OldProposalFreshSample(doc);
    }

    int32_t LightDocSampler::InferOneDoc(LDADocument *doc)
    {
        return OldProposalFreshSampleInfer(doc);
    }
    int32_t LightDocSampler::Sample2WordFirst(LDADocument *doc, int32_t w, int32_t s, int32_t old_topic)
    {
        int32_t w_t_cnt;
        int32_t w_s_cnt;

        real_t n_td_alpha;
        real_t n_sd_alpha;
        real_t n_tw_beta;
        real_t n_sw_beta;
        real_t n_s_beta_sum;
        real_t n_t_beta_sum;

        real_t proposal_s;
        real_t proposal_t;

        real_t nominator;
        real_t denominator;

        real_t rejection;
        real_t pi;
        int m;

        for (int i = 0; i < mh_step_for_gs_; ++i)
        {
            int32_t t;

            t = alias_k_v_[w].next(rng_, beta_height_, beta_mass_, beta_k_v_, false);

            rejection = rng_.rand_real();

            n_td_alpha = doc_topic_counter_[t] + alpha_;
            n_sd_alpha = doc_topic_counter_[s] + alpha_;


            w_s_cnt = get_word_topic(w, s);
            w_t_cnt = get_word_topic(w, t);

            if (s != old_topic && t != old_topic)
            {
                n_tw_beta = w_t_cnt + beta_;
                n_t_beta_sum = summary_row_[t] + beta_sum_;

                n_sw_beta = w_s_cnt + beta_;
                n_s_beta_sum = summary_row_[s] + beta_sum_;
            }
            else if (s != old_topic && t == old_topic)
            {
                n_td_alpha -= 1;

                n_tw_beta = w_t_cnt + beta_ - 1;
                n_t_beta_sum = summary_row_[t] + beta_sum_ - 1;

                n_sw_beta = w_s_cnt + beta_;
                n_s_beta_sum = summary_row_[s] + beta_sum_;
            }
            else if (s == old_topic && t != old_topic)
            {
                n_sd_alpha -= 1;

                n_tw_beta = w_t_cnt + beta_;
                n_t_beta_sum = summary_row_[t] + beta_sum_;

                n_sw_beta = w_s_cnt + beta_ - 1;
                n_s_beta_sum = summary_row_[s] + beta_sum_ - 1;
            }
            else
            {
                n_td_alpha -= 1;
                n_sd_alpha -= 1;

                n_tw_beta = w_t_cnt + beta_ - 1;
                n_t_beta_sum = summary_row_[t] + beta_sum_ - 1;

                n_sw_beta = w_s_cnt + beta_ - 1;
                n_s_beta_sum = summary_row_[s] + beta_sum_ - 1;
            }

            proposal_s = (w_s_cnt + beta_) / (summary_row_[s] + beta_sum_);
            proposal_t = (w_t_cnt + beta_) / (summary_row_[t] + beta_sum_);

            nominator = n_td_alpha
                * n_tw_beta
                * n_s_beta_sum
                * proposal_s;

            denominator = n_sd_alpha
                * n_sw_beta
                * n_t_beta_sum
                * proposal_t;


            pi = std::min<real_t>((real_t)1.0, nominator / denominator);

            // s = rejection < pi ? t : s;
            m = -(rejection < pi);
            s = (t & m) | (s & ~m);

            real_t n_td_or_alpha = rng_.rand_real() * (n_td_sum_ + alpha_sum_);
            if (n_td_or_alpha < n_td_sum_)
            {
                int32_t t_idx = rng_.rand_k(doc_size_);
                t = doc->Topic(t_idx);
            }
            else
            {
                t = rng_.rand_k(K_);
            }

            rejection = rng_.rand_real();

            n_td_alpha = doc_topic_counter_[t] + alpha_;
            n_sd_alpha = doc_topic_counter_[s] + alpha_;


            if (s != old_topic && t != old_topic)
            {
                w_t_cnt = get_word_topic(w, t);
                n_tw_beta = w_t_cnt + beta_;
                n_t_beta_sum = summary_row_[t] + beta_sum_;

                w_s_cnt = get_word_topic(w, s);
                n_sw_beta = w_s_cnt + beta_;
                n_s_beta_sum = summary_row_[s] + beta_sum_;
            }
            else if (s != old_topic && t == old_topic)
            {
                n_td_alpha -= 1;

                w_t_cnt = get_word_topic(w, t) - 1;
                n_tw_beta = w_t_cnt + beta_;
                n_t_beta_sum = summary_row_[t] + beta_sum_ - 1;

                w_s_cnt = get_word_topic(w, s);
                n_sw_beta = w_s_cnt + beta_;
                n_s_beta_sum = summary_row_[s] + beta_sum_;
            }
            else if (s == old_topic && t != old_topic)
            {
                n_sd_alpha -= 1;

                w_t_cnt = get_word_topic(w, t);
                n_tw_beta = w_t_cnt + beta_;
                n_t_beta_sum = summary_row_[t] + beta_sum_;

                w_s_cnt = get_word_topic(w, s) - 1;
                n_sw_beta = w_s_cnt + beta_;
                n_s_beta_sum = summary_row_[s] + beta_sum_ - 1;
            }
            else
            {
                n_td_alpha -= 1;
                n_sd_alpha -= 1;

                w_t_cnt = get_word_topic(w, t) - 1;
                n_tw_beta = w_t_cnt + beta_;
                n_t_beta_sum = summary_row_[t] + beta_sum_ - 1;

                w_s_cnt = get_word_topic(w, s) - 1;
                n_sw_beta = w_s_cnt + beta_;
                n_s_beta_sum = summary_row_[s] + beta_sum_ - 1;
            }

            proposal_t = doc_topic_counter_[t] + alpha_;
            proposal_s = doc_topic_counter_[s] + alpha_;

            nominator = n_td_alpha
                * n_tw_beta
                * n_s_beta_sum
                * proposal_s;

            denominator = n_sd_alpha
                * n_sw_beta
                * n_t_beta_sum
                * proposal_t;


            pi = std::min<real_t>((real_t)1.0, nominator / denominator);

            // s = rejection < pi ? t : s;
            m = -(rejection < pi);
            s = (t & m) | (s & ~m);
        }
        int32_t src = s;
        return src;
    }

    int32_t LightDocSampler::Sample2WordFirstInfer(LDADocument *doc, int32_t w, int32_t s, int32_t old_topic)
    {
        int32_t w_t_cnt;
        int32_t w_s_cnt;

        float n_td_alpha;
        float n_sd_alpha;
        float n_tw_beta;
        float n_sw_beta;
        float n_s_beta_sum;
        float n_t_beta_sum;

        /*float proposal_s;
        float proposal_t;*/

        float nominator;
        float denominator;

        float rejection;
        float pi;
        int m;

        for (int i = 0; i < mh_step_for_gs_; ++i)
        {
            int32_t t;
            // n_tw proposal
            t = alias_k_v_[w].next(rng_, beta_height_, beta_mass_, beta_k_v_, false);

            rejection = rng_.rand_real();

            n_td_alpha = doc_topic_counter_[t] + alpha_;
            n_sd_alpha = doc_topic_counter_[s] + alpha_;

            nominator = n_td_alpha;
            denominator = n_sd_alpha;

            pi = std::min((float)1.0, nominator / denominator);

            //s = rejection < pi ? t : s;
            m = -(rejection < pi);
            s = (t & m) | (s & ~m);

            float n_td_or_alpha = rng_.rand_real() * (n_td_sum_ + alpha_sum_);
            if (n_td_or_alpha < n_td_sum_)
            {
                int32_t t_idx = rng_.rand_k(doc_size_);
                t = doc->Topic(t_idx);
            }
            else
            {
                t = rng_.rand_k(K_);
            }

            rejection = rng_.rand_real();


            w_t_cnt = get_word_topic(w, t);
            n_tw_beta = w_t_cnt + beta_;
            n_t_beta_sum = summary_row_[t] + beta_sum_;

            w_s_cnt = get_word_topic(w, s);
            n_sw_beta = w_s_cnt + beta_;
            n_s_beta_sum = summary_row_[s] + beta_sum_;

            nominator = n_tw_beta
                * n_s_beta_sum;


            denominator = n_sw_beta
                * n_t_beta_sum;

            pi = std::min((float)1.0, nominator / denominator);

            //s = rejection < pi ? t : s;
            m = -(rejection < pi);
            s = (t & m) | (s & ~m);
        }
        int32_t src = s;
        return src;
    }

    int32_t LightDocSampler::OldProposalFreshSample(LDADocument *doc)
    {
        DocInit(doc);
        int num_token = doc->size();
        int32_t &cursor = doc->get_cursor();

        int32_t token_sweeped = 0;
        cursor = 0;

        while (cursor < num_token)
        {
            ++token_sweeped;

            int32_t w = doc->Word(cursor);
            int32_t s = doc->Topic(cursor);            // old topic

            int t = Sample2WordFirst(doc, w, s, s);    // new topic

            if (s != t)
            {
                word_topic_delta wtd;
                int32_t shard_id = w % num_threads_;
                wtd.word = w;
                wtd.topic = s;
                wtd.delta = -1;
                word_topic_delta_[shard_id].push_back(wtd);

                wtd.topic = t;
                wtd.delta = +1;
                word_topic_delta_[shard_id].push_back(wtd);

                --delta_summary_row_[s];
                ++delta_summary_row_[t];

                doc->SetTopic(cursor, t);
                doc_topic_counter_.inc(s, -1);
                doc_topic_counter_.inc(t, 1);
            }
            cursor++;
        }
        return token_sweeped;
    }

    int32_t LightDocSampler::OldProposalFreshSampleInfer(LDADocument *doc)
    {

        DocInit(doc);
        int num_token = doc->size();
        int32_t &cursor = doc->get_cursor();

        int32_t token_sweeped = 0;
        cursor = 0;

        while (cursor < num_token)
        {
            ++token_sweeped;

            int32_t w = doc->Word(cursor);
            int32_t s = doc->Topic(cursor);            // old topic

            int t = Sample2WordFirstInfer(doc, w, s, s);    // new topic

            if (s != t)
            {
                doc->SetTopic(cursor, t);
                doc_topic_counter_.inc(s, -1);
                doc_topic_counter_.inc(t, 1);
            }
            cursor++;
        }
        return token_sweeped;
    }

    double LightDocSampler::NormalizeWordLLH()
    {
        double word_llh = K_ * log_topic_normalizer_;
        for (int k = 0; k < K_; ++k)
        {
            word_llh -= LogGamma(summary_row_[k] + beta_sum_);
        }
        return word_llh;
    }


    double LightDocSampler::ComputeOneDocLLH(LDADocument* doc)
    {
        double doc_ll = 0;
        double one_doc_llh = log_doc_normalizer_;

        // Compute doc-topic vector on the fly.
        int num_tokens = doc->size();

        if (num_tokens == 0)
        {
            return doc_ll;
        }

        doc_topic_counter_.clear();
        doc->GetDocTopicCounter(doc_topic_counter_);

        int32_t capacity = doc_topic_counter_.capacity();
        int32_t *key = doc_topic_counter_.key();
        int32_t *value = doc_topic_counter_.value();
        int32_t nonzero_num = 0;

        for (int i = 0; i < capacity; ++i)
        {
            if (key[i] > 0)
            {
                one_doc_llh += LogGamma(value[i] + ll_alpha_);
                ++nonzero_num;
            }
        }
        one_doc_llh += (K_ - nonzero_num) * LogGamma(ll_alpha_);
        one_doc_llh -= LogGamma(num_tokens + ll_alpha_ * K_);

        // CHECK_EQ(one_doc_llh, one_doc_llh) << "one_doc_llh is nan.";

        doc_ll += one_doc_llh;
        return doc_ll;
    }

    double LightDocSampler::ComputeWordLLH(int32_t lower, int32_t upper)
    {
        // word_llh is P(w|z).
        double word_llh = 0;
        double zero_entry_llh = LogGamma(beta_);

        // Since some vocabs are not present in the corpus, use num_words_seen to
        // count # of words in corpus.
        int num_words_seen = 0;
        for (int w = lower; w < upper; ++w)
        {
            auto word_topic_row = get_word_row(w);
            int32_t total_count = 0;
            double delta = 0;
            if (word_topic_row.is_dense())
            {
                int32_t* memory = word_topic_row.memory();
                int32_t capacity = word_topic_row.capacity();
                int32_t count;
                for (int i = 0; i < capacity; ++i)
                {
                    count = memory[i];
                    // CHECK_LE(0, count) << "negative count. " << count;
                    total_count += count;
                    delta += LogGamma(count + beta_);
                }
            }
            else
            {
                int32_t* key = word_topic_row.key();
                int32_t* value = word_topic_row.value();
                int32_t capacity = word_topic_row.capacity();
                int32_t count;
                int32_t nonzero_num = 0;
                for (int i = 0; i < capacity; ++i)
                {
                    if (key[i] > 0)
                    {
                        count = value[i];
                        // CHECK_LE(0, count) << "negative count. " << count;
                        total_count += count;
                        delta += LogGamma(count + beta_);
                        ++nonzero_num;
                    }
                }
                delta += (K_ - nonzero_num) * zero_entry_llh;
            }

            if (total_count)
            {
                word_llh += delta;
            }
        }

        // CHECK_EQ(word_llh, word_llh) << "word_llh is nan.";
        return word_llh;
    }

    void LightDocSampler::Dump(const std::string &dump_name, int32_t lower, int32_t upper)
    {
        std::ofstream wt_stream;
        wt_stream.open(dump_name, std::ios::out);
        // CHECK(wt_stream.good()) << "Open word_topic_dump file: " << dump_name;

        for (int w = lower; w < upper; ++w)
        {
            //taifeng: why not just a serialization of current hybrid_map? do we need to do a search?
            int nonzero_num = word_topic_table_[w].nonzero_num();
            if (nonzero_num)
            {
                wt_stream << w;
                for (int t = 0; t < K_; ++t)
                {
                    // if (word_topic_table_[w * K_ + t])
                    // if (word_topic_table_[(int64_t)w * K_ + t])
                    if (word_topic_table_[w][t] > 0)
                    {
                        wt_stream << " " << t << ":" << word_topic_table_[w][t];
                    }
                }
                wt_stream << std::endl;
            }
        }
        wt_stream.close();
    }
}
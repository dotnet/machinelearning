#pragma once

#include "type_common.h"
#include "lda_document.h"
#include "rand_int_rng.h"
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <queue>
#include <map>
#include "alias_multinomial_rng_int.hpp"
#include "light_hash_map.h"
#include "utils.hpp"
#include "hybrid_map.h"
#include "hybrid_alias_map.h"
#include "model_block.h"

namespace lda
{
	struct word_topic_delta
	{
		int32_t word;
		int32_t topic;
		int32_t delta;
	};

	class LightDocSampler
	{
	public:
		LightDocSampler(
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
			float &beta_mass,
			std::vector<wood::alias_k_v> &beta_k_v
			);

		~LightDocSampler();

		int32_t GlobalInit(LDADocument *doc);
		int32_t DocInit(LDADocument *doc);
		void EpocInit();
		void AdaptAlphaSum(bool is_train);
		void GetDocTopic(LDADocument *doc, int* pTopics, int* pProbs, int32_t& numTopicsMax);


		int32_t SampleOneDoc(LDADocument *doc);
		int32_t InferOneDoc(LDADocument *doc);
		
		// The i-th complete-llh calculation will use row i in llh_able_. This is
		// part of log P(z) in eq.[3].
		double ComputeOneDocLLH(LDADocument* doc);
		double ComputeWordLLH(int32_t lower, int32_t upper);
		double NormalizeWordLLH();

		inline void rng_restart()
		{
			rng_.restart();
		}


		void Dump(const std::string &dump_name, int32_t lower, int32_t upper);

		void build_alias_table(int32_t lower, int32_t upper, int thread_id);
		void build_word_topic_table(int32_t thread_id, int32_t num_threads, lda::LDAModelBlock &model_block);

		inline int32_t rand_k();
		inline wood::xorshift_rng& rng();
		inline lda::hybrid_map& get_word_row(int32_t word);
		inline std::vector<int64_t> &get_summary_row();
		inline std::vector<word_topic_delta>& get_word_topic_delta(int32_t thread_id);
		inline std::vector<int64_t>& get_delta_summary_row();

	private:
		int32_t Sample2WordFirst(LDADocument *doc, int32_t w, int32_t s, int32_t old_topic);
		int32_t Sample2WordFirstInfer(LDADocument *doc, int32_t w, int32_t s, int32_t old_topic);
		inline void GenerateAliasTableforWord(int32_t word);
		inline int32_t get_word_topic(int32_t word, int32_t topic);
		inline void word_topic_dec(int32_t word, int32_t topic);
		inline void word_topic_inc(int32_t word, int32_t topic);
		int32_t OldProposalFreshSample(LDADocument *doc);
		int32_t OldProposalFreshSampleInfer(LDADocument *doc);

	private:
		int32_t num_tokens_;
		int32_t num_unique_words_;

		int32_t K_;
		int32_t V_;
		real_t beta_;
		real_t beta_sum_;
		real_t alpha_;
		real_t alpha_sum_;

		real_t ll_alpha_;
		real_t ll_alpha_sum_;

		real_t delta_alpha_sum_;

		std::vector<float> q_w_proportion_;
		wood::AliasMultinomialRNGInt alias_rng_;
		wood::xorshift_rng rng_;
		std::vector<lda::hybrid_alias_map> &alias_k_v_;

		int32_t doc_size_;

		// the number of Metropolis Hastings step
		int32_t mh_step_for_gs_;
		real_t n_td_sum_;
		
		// model
		std::vector<int64_t> &summary_row_;
		std::vector<lda::hybrid_map> &word_topic_table_;
		int32_t *rehashing_buf_;

		int32_t &beta_height_;
		float &beta_mass_;
		std::vector<wood::alias_k_v> &beta_k_v_;

		// delta
		std::vector<int64_t> delta_summary_row_;

		int32_t num_threads_;
		std::vector<std::vector<word_topic_delta>> word_topic_delta_;	

		// ================ Precompute LLH Parameters =================
		// Log of normalization constant (per docoument) from eq.[3].
		double log_doc_normalizer_;

		// Log of normalization constant (per topic) from eq.[2].
		double log_topic_normalizer_;
		lda::light_hash_map doc_topic_counter_;
	};
	
	inline int32_t LightDocSampler::rand_k()
	{
		return rng_.rand_k(K_);
	}
	inline wood::xorshift_rng& LightDocSampler::rng()
	{
		return rng_;
	}
	inline lda::hybrid_map& LightDocSampler::get_word_row(int32_t word)
	{
		return word_topic_table_[word];
	}
	inline std::vector<int64_t>& LightDocSampler::get_summary_row()
	{
		return summary_row_;
	}
	inline std::vector<word_topic_delta>& LightDocSampler::get_word_topic_delta(int32_t thread_id)
	{
		return word_topic_delta_[thread_id];
	}
	inline std::vector<int64_t>& LightDocSampler::get_delta_summary_row()
	{
		return delta_summary_row_;
	}
	inline int32_t LightDocSampler::get_word_topic(int32_t word, int32_t topic)
	{
		return word_topic_table_[word][topic];
	}	
	inline void LightDocSampler::word_topic_dec(int32_t word, int32_t topic)
	{
		word_topic_table_[word].inc(topic, -1);
	}
	inline void LightDocSampler::word_topic_inc(int32_t word, int32_t topic)
	{
		word_topic_table_[word].inc(topic, 1);
	}
	inline void LightDocSampler::GenerateAliasTableforWord(int32_t word)
	{
		alias_k_v_[word].build_table(alias_rng_, word_topic_table_[word], summary_row_, q_w_proportion_, beta_, beta_sum_, word, rng_);
	}
}
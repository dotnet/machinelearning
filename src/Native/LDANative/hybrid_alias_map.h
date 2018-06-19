// author: jiyuan
// date  : 2014.9.15

#pragma once
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <cassert>
#include <map>
#include "alias_multinomial_rng_int.hpp"
#include "hybrid_map.h"

namespace lda
{
	class hybrid_alias_map
	{
	public:

		hybrid_alias_map();
		hybrid_alias_map(int32_t *memory, int32_t is_dense, int32_t capacity);
		hybrid_alias_map(const hybrid_alias_map &other);
		hybrid_alias_map& operator=(const hybrid_alias_map &other);

		void clear();
		inline int32_t size() const;

		std::string DebugString();
		void build_table(
			wood::AliasMultinomialRNGInt &alias_rng,
			const hybrid_map &word_topic_row,
			const std::vector<int64_t> &summary_row,
			std::vector<float> &q_w_proportion,
			float beta,
			float beta_sum,
			int word_id,
			wood::xorshift_rng &rng);
	
		inline int32_t next(wood::xorshift_rng &rng, int32_t beta_height, float beta_mass, std::vector<wood::alias_k_v> &beta_k_v, bool debug);

	private:
		int32_t *memory_;
		int32_t is_dense_;
		int32_t *kv_;
		int32_t *idx_;
		int32_t height_;
		int32_t capacity_;
		int32_t size_;

		float mass_;
		float n_kw_mass_;
		float beta_mass_;
		// static std::vector<wood::alias_k_v> beta_kv_;
	};

	inline int32_t hybrid_alias_map::size() const
	{
		return size_;
	}

	inline int32_t hybrid_alias_map::next(wood::xorshift_rng &rng, int32_t beta_height, float beta_mass, std::vector<wood::alias_k_v> &beta_k_v, bool debug)
	{
		//note(taifengw), here we will set those unseen words' topic to 0. logicall we could set it to random as well.
		if (capacity_ == 0)
		{
			return 0;
		}

		if (is_dense_)
		{
			auto sample = rng.rand();
			int idx = sample / height_;
			if (idx >= size_)
			{
				idx = size_ - 1;
			}

			int32_t *p = memory_ + 2 * idx;
			int32_t k = *p;
			p++;
			int32_t v = *p;
			int32_t m = -(sample < v);
			return (idx & m) | (k & ~m);
		}
		else
		{
			float sample = rng.rand_real() * (n_kw_mass_ + beta_mass);
			if (sample < n_kw_mass_)
			{
				auto n_kw_sample = rng.rand();
				int32_t idx = n_kw_sample / height_;

				if (idx >= size_)
				{
					idx = size_ - 1;
				}


				int32_t *p = memory_ + 2 * idx;
				int32_t k = *p; p++;
				int32_t v = *p;
				int32_t id = idx_[idx];
				int32_t k_id = idx_[k];

				// return n_kw_sample < v ? id : k_id;
				int32_t m = -(n_kw_sample < v);
				return (id & m) | (k_id & ~m);

			}
			else
			{
				auto sampleLocal = rng.rand();
				int idx = sampleLocal / beta_height;
				int beta_size = (int)beta_k_v.size();

				if (idx >= beta_size)
				{
					idx = beta_size - 1;
				}

				int32_t k = beta_k_v[idx].k_;
				int32_t v = beta_k_v[idx].v_;
				int32_t m = -(sampleLocal < v);
				return (idx & m) | (k & ~m);
			}
		}
	}

}
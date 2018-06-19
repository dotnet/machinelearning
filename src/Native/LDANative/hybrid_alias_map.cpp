
#include <fstream>
#include <iostream>
#include <vector>
#include "utils.hpp"
#include <stdlib.h>
#include "hybrid_alias_map.h"

namespace lda
{
	hybrid_alias_map::hybrid_alias_map()
		:memory_(nullptr),
		is_dense_(1),
		kv_(nullptr),
		idx_(nullptr),
		capacity_(0),
		size_(0),
		mass_(0),
		n_kw_mass_(0.0),
		beta_mass_(0.0)
	{
		// CHECK(is_dense_) << "is_dense_ == 0";
	}
	hybrid_alias_map::hybrid_alias_map(int32_t *memory, int32_t is_dense, int32_t capacity)
		:memory_(memory),
		is_dense_(is_dense),
		capacity_(capacity),
		kv_(nullptr),
		idx_(nullptr),
		size_(0),
		mass_(0),
		n_kw_mass_(0.0),
		beta_mass_(0.0)
	{
		if (is_dense_)
		{
			kv_ = memory_;
			idx_ = nullptr;
		}
		else
		{
			kv_ = memory_;
			idx_ = memory_ + capacity_ * 2;
		}
	}

	hybrid_alias_map::hybrid_alias_map(const hybrid_alias_map &other)
	{
		this->memory_ = other.memory_;
		this->is_dense_ = other.is_dense_;
		this->capacity_ = other.capacity_;

		this->kv_ = other.kv_;
		this->idx_ = other.idx_;
		this->height_ = other.height_;
		this->size_ = other.size_;

		this->mass_ = other.mass_;
		this->n_kw_mass_ = other.n_kw_mass_;
		this->beta_mass_ = other.beta_mass_;
	}
	hybrid_alias_map& hybrid_alias_map::operator=(const hybrid_alias_map &other)
	{
		this->memory_ = other.memory_;
		this->is_dense_ = other.is_dense_;
		this->capacity_ = other.capacity_;

		this->kv_ = other.kv_;
		this->idx_ = other.idx_;
		this->height_ = other.height_;
		this->size_ = other.size_;

		this->mass_ = other.mass_;
		this->n_kw_mass_ = other.n_kw_mass_;
		this->beta_mass_ = other.beta_mass_;

		return *this;
	}

	void hybrid_alias_map::clear()
	{
		size_ = 0;
	}

	std::string hybrid_alias_map::DebugString()
	{
		std::string str = "";

		if (size_ == 0)
		{
			return str;
		}

		str += "is_dense:" + std::to_string(is_dense_) + " height:" + std::to_string(height_) + " mass:" + std::to_string(n_kw_mass_);
		if (is_dense_)
		{
			for (int i = 0; i < capacity_; ++i)
			{
				str += " " + std::to_string(i) + ":" + std::to_string(*(memory_ + 2 * i)) + ":" + std::to_string(*(memory_ + 2 * i + 1));
			}
		}
		else
		{
			for (int i = 0; i < size_; ++i)
			{
				str += " " + std::to_string(idx_[i]) + ":" + std::to_string(*(memory_ + 2 * i)) + ":" + std::to_string(*(memory_ + 2 * i + 1));
			}
		}

		return str;
	}

	void hybrid_alias_map::build_table(
		wood::AliasMultinomialRNGInt &alias_rng,
		const hybrid_map &word_topic_row,
		const std::vector<int64_t> &summary_row,
		std::vector<float> &q_w_proportion,
		float beta,
		float beta_sum,
		int word_id,
		wood::xorshift_rng &rng)
	{
		if (is_dense_)
		{
			size_ = capacity_;
			mass_ = 0.0;
			for (int k = 0; k < capacity_; ++k)
			{
				int32_t n_kw = word_topic_row[k];
				float prop = (n_kw + beta) / (summary_row[k] + beta_sum);
				q_w_proportion[k] = prop;
				mass_ += prop;
			}
			if (size_ == 0)
			{
				return;
			}
			alias_rng.SetProportionMass(q_w_proportion, mass_, memory_, &height_, rng);

		}
		else
		{
			if (word_topic_row.is_dense())
			{
				size_ = 0;
				n_kw_mass_ = 0.0;
				for (int k = 0; k < word_topic_row.capacity_; ++k)
				{
					if (word_topic_row.memory_[k] == 0) continue;
					int32_t n_tw = word_topic_row.memory_[k];
					int64_t n_t = summary_row[k];
					q_w_proportion[size_] = n_tw / (n_t + beta_sum);
					idx_[size_] = k;
					n_kw_mass_ += q_w_proportion[size_];
					++size_;
				}

				if (size_ == 0)
				{
					// it is possible that, the local tf of a word is zero
					return;
				}
				alias_rng.SetProportionMass(q_w_proportion, size_, n_kw_mass_, memory_, &height_, rng, word_id);
			}
			else
			{
				size_ = 0;
				n_kw_mass_ = 0.0;
				int32_t row_capacity = word_topic_row.capacity_;
				for (int k = 0; k < row_capacity; ++k)
				{
					int32_t key = word_topic_row.key_[k];
					if (key > 0)
					{
						int32_t n_kw = word_topic_row.value_[k];
						float prop = n_kw / (summary_row[key - 1] + beta_sum);



						q_w_proportion[size_] = prop;
						idx_[size_] = word_topic_row.key_[k] - 1;   // minus one from the the internal key
						n_kw_mass_ += prop;

						++size_;
					}
				}
				if (size_ == 0)
				{
					// it is possible that, the local tf of a word is zero
					return;
				}
				alias_rng.SetProportionMass(q_w_proportion, size_, n_kw_mass_, memory_, &height_, rng, word_id);
			}
		}
	}
}

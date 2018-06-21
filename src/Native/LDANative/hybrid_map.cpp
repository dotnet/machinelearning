// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#include <fstream>
#include <iostream>
#include <vector>
#include "utils.hpp"
#include <stdlib.h>
#include "hybrid_map.h"

namespace lda
{
	hybrid_map::hybrid_map()
		:memory_(nullptr),
		is_dense_(1),
		capacity_(0),
		empty_key_(0),
		deleted_key_(-1),
		key_(nullptr),
		value_(nullptr),
		num_deleted_key_(0),
		external_rehash_buf_(nullptr)
	{
		// CHECK(is_dense_) << "is_dense_ == 0";
	}
	hybrid_map::hybrid_map(int32_t *memory, int32_t is_dense, int32_t capacity, int32_t num_deleted_key
		, int32_t *external_rehash_buf_)
		: memory_(memory),
		is_dense_(is_dense),
		capacity_(capacity),
		empty_key_(0),
		deleted_key_(-1),
		key_(nullptr),
		value_(nullptr),
		num_deleted_key_(num_deleted_key),
		external_rehash_buf_(external_rehash_buf_)
	{
		if (is_dense_ == 0) {
			key_ = memory_;
			value_ = memory_ + capacity_;
		}
	}

	hybrid_map::hybrid_map(const hybrid_map &other)
	{
		this->memory_ = other.memory_;
		this->is_dense_ = other.is_dense_;
		this->capacity_ = other.capacity_;
		empty_key_ = other.empty_key_;
		deleted_key_ = other.deleted_key_;
		num_deleted_key_ = other.num_deleted_key_;
		external_rehash_buf_ = other.external_rehash_buf_;
		if (this->is_dense_)
		{
			this->key_ = nullptr;
			this->value_ = nullptr;
		}
		else
		{
			this->key_ = this->memory_;
			this->value_ = this->memory_ + capacity_;
		}

	}
	hybrid_map& hybrid_map::operator=(const hybrid_map &other)
	{
		this->memory_ = other.memory_;
		this->is_dense_ = other.is_dense_;
		this->capacity_ = other.capacity_;
		empty_key_ = other.empty_key_;
		deleted_key_ = other.deleted_key_;
		num_deleted_key_ = other.num_deleted_key_;
		external_rehash_buf_ = other.external_rehash_buf_;
		if (this->is_dense_)
		{
			this->key_ = nullptr;
			this->value_ = nullptr;
		}
		else
		{
			this->key_ = this->memory_;
			this->value_ = this->memory_ + capacity_;
		}
		return *this;
	}

	void hybrid_map::clear()
	{
		int32_t memory_size = is_dense_ ? capacity_ : 2 * capacity_;
		memset(memory_, 0, memory_size * sizeof(int32_t));
	}

	std::string hybrid_map::DumpString() const
	{
		if (is_dense_)
		{
			std::string result;
			for (int i = 0; i < capacity_; ++i)
			{
				if (memory_[i] != 0)
				{
					result += std::to_string(i) + ":" + std::to_string(memory_[i]) + " ";
				}
			}
			return result;
		}
		else
		{
			std::string result;
			for (int i = 0; i < capacity_; ++i)
			{
				if (key_[i] > 0)
				{
					result += std::to_string(key_[i] - 1) + ":" + std::to_string(value_[i]) + " ";
				}
			}
			return result;
		}
	}

	void hybrid_map::sorted_rehashing()
	{
		if (!is_dense_)
		{
			std::map<int32_t, int32_t> rehash_buffer;
			for (int i = 0; i < capacity_; ++i)
			{
				if (key_[i] > 0)
				{
					rehash_buffer[key_[i] - 1] = value_[i];
				}
			}
			memset(memory_, 0, 2 * capacity_ * sizeof(int32_t));
			for (auto it = rehash_buffer.begin();
				it != rehash_buffer.end(); ++it)
			{
				inc(it->first, it->second);
			}
		}
	}

}

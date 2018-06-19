#include <cstring>
#include "light_hash_map.h"

namespace lda
{
	light_hash_map::light_hash_map(int32_t *mem_block, int32_t capacity) :
		own_memory_(false),
		capacity_(capacity),
		mem_block_(mem_block),
		empty_key_(0),
		deleted_key_(-2)
	{
		key_ = mem_block_;
		value_ = mem_block_ + capacity_;
		clear();
	}

	light_hash_map::light_hash_map(int32_t capacity) :
		own_memory_(true),
		capacity_(capacity),
		empty_key_(0),
		deleted_key_(-2)
	{
		mem_block_ = new int32_t[capacity_ * 2];
		key_ = mem_block_;
		value_ = mem_block_ + capacity_;
		clear();
	}

	// must call set_memory after construction before use
	light_hash_map::light_hash_map() :
		capacity_(1024),
		own_memory_(false),
		empty_key_(0),
		deleted_key_(-2),
		mem_block_(nullptr),
		key_(nullptr),
		value_(nullptr)
	{
	}

	light_hash_map::~light_hash_map()
	{
		capacity_ = 0;
		if (own_memory_ && mem_block_ != nullptr)
		{
			delete[]mem_block_;
		}

		mem_block_ = nullptr;
		key_ = nullptr;
		value_ = nullptr;
	}

	void light_hash_map::clear()
	{
		memset(mem_block_, 0, capacity_ * 2 * sizeof(int32_t));
	}

	void light_hash_map::sort()
	{
		//key is probablly empty in key_, sort by value_
		//this is just for the output process like getting the topic of document or a topic of term
	}

	void light_hash_map::set_memory(int32_t *mem_block)
	{
		mem_block_ = mem_block;
		key_ = mem_block_;
		value_ = mem_block_ + capacity_;
	}
}
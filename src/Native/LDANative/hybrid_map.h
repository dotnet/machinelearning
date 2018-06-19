// author: jiyuan
// date  : 2014.10.1

#pragma once
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <cassert>
#include <map>

// The probing method:
// Linear probing
// #define JUMP_(key, num_probes)    ( 1 )

// Quadratic probing
#define JUMP_(key, num_probes)    ( num_probes )
#define ILLEGAL_BUCKET -1

namespace lda
{	
	class hybrid_alias_map;

	class hybrid_map
	{
		friend class hybrid_alias_map;
	public:
		hybrid_map();
		hybrid_map(int32_t *memory, int32_t is_dense, int32_t capacity, int32_t num_deleted_key
			, int32_t *external_rehash_buf_);
		hybrid_map(const hybrid_map &other);
		hybrid_map& operator=(const hybrid_map &other);


		void clear();
		std::string DumpString() const;
		void sorted_rehashing();

		inline int32_t nonzero_num() const;
		inline bool is_dense() const;
		inline int32_t capacity() const;
		inline int32_t *memory() const;
		inline int32_t* key() const;
		inline int32_t* value() const;
		inline void rehashing();
		inline void inc(int32_t key, int32_t delta);
		// query the value of |key|
		// if |key| is in the table, return the |value| corresonding to |key|
		// if not, just return 0
		inline int32_t operator[](int32_t key) const;
	
	private:
		inline std::pair<int32_t, int32_t> find_position(const int32_t key) const;

		int32_t *memory_;
		int32_t is_dense_;
		int32_t *key_;
		int32_t *value_;

		// if |is_dense_| == true, capactiy_ is the length of an array
		// if |is dense_| == false, capacity_ is the size of a light hash table
		int32_t capacity_;
		int32_t empty_key_;
		int32_t deleted_key_;

		int32_t num_deleted_key_;
		int32_t* external_rehash_buf_;
	};

	inline int32_t hybrid_map::nonzero_num() const
	{
		if (is_dense_)
		{
			int32_t size = 0;
			for (int i = 0; i < capacity_; ++i)
			{
				if (memory_[i] > 0)
				{
					++size;
				}
			}
			return size;
		}
		else
		{
			int32_t size = 0;
			for (int i = 0; i < capacity_; ++i)
			{
				if (key_[i] > 0)
				{
					++size;
				}
			}
			return size;
		}
	}

	inline bool hybrid_map::is_dense() const
	{
		return is_dense_ != 0;
	}

	inline int32_t hybrid_map::capacity() const
	{
		return capacity_;
	}

	inline int32_t* hybrid_map::memory() const
	{
		return memory_;
	}
	inline int32_t* hybrid_map::key() const
	{
		return key_;
	}
	inline int32_t* hybrid_map::value() const
	{
		return value_;
	}
	inline void hybrid_map::rehashing()
	{
		if (!is_dense_)
		{
			memcpy(external_rehash_buf_, memory_, 2 * capacity_ * sizeof(int32_t));
			int32_t *key = external_rehash_buf_;
			int32_t *value = external_rehash_buf_ + capacity_;
			memset(memory_, 0, 2 * capacity_ * sizeof(int32_t));
			for (int i = 0; i < capacity_; ++i)
			{
				if (key[i] > 0)
				{
					inc(key[i] - 1, value[i]);
				}
			}
			num_deleted_key_ = 0;
		}
	}
	inline void hybrid_map::inc(int32_t key, int32_t delta)
	{
		if (is_dense_)
		{
			memory_[key] += delta;
		}
		else
		{
			int32_t internal_key = key + 1;
			std::pair<int32_t, int32_t> pos = find_position(internal_key);
			if (pos.first != ILLEGAL_BUCKET)
			{
				value_[pos.first] += delta;
				if (value_[pos.first] == 0)       // the value becomes zero, delete the key
				{
					key_[pos.first] = deleted_key_;

					++num_deleted_key_;        // num_deleted_key ++
					if (num_deleted_key_ * 20 > capacity_)
					{
						rehashing();
					}
				}
			}
			else                                 // not found the key, insert it with delta as value
			{
				key_[pos.second] = internal_key;
				value_[pos.second] = delta;
			}
		}
	}

	// query the value of |key|
	// if |key| is in the table, return the |value| corresonding to |key|
	// if not, just return 0
	inline int32_t hybrid_map::operator[](int32_t key) const
	{
		if (is_dense_)
		{
			//return memory_[key];
			if (capacity_ > 0)
			{
				return memory_[key];
			}
			else
			{
				return 0;
			}
		}
		else
		{
			int32_t internal_key = key + 1;
			std::pair<int32_t, int32_t> pos = find_position(internal_key);
			if (pos.first != ILLEGAL_BUCKET)
			{
				return value_[pos.first];
			}
			else
			{
				return 0;
			}
		}
	}
	inline std::pair<int32_t, int32_t> hybrid_map::find_position(const int32_t key) const
	{
		int num_probes = 0;
		int32_t capacity_minus_one = capacity_ - 1;
		//int32_t idx = hasher_(key) & capacity_minus_one;
		int32_t idx = key % capacity_;
		int32_t insert_pos = ILLEGAL_BUCKET;
		while (1)                                           // probe until something happens
		{
			if (key_[idx] == empty_key_)                    // bucket is empty
			{
				if (insert_pos == ILLEGAL_BUCKET)           // found no prior place to insert
				{
					return std::pair<int32_t, int32_t>(ILLEGAL_BUCKET, idx);
				}
				else                                        // previously, there is a position to insert
				{
					return std::pair<int32_t, int32_t>(ILLEGAL_BUCKET, insert_pos);
				}
			}
			else if (key_[idx] == deleted_key_)            // keep searching, but makr to insert
			{
				if (insert_pos == ILLEGAL_BUCKET)
				{
					insert_pos = idx;
				}
			}
			else if (key_[idx] == key)
			{
				return std::pair<int32_t, int32_t>(idx, ILLEGAL_BUCKET);
			}
			++num_probes;                                // we are doing another probe
			idx = (idx + JUMP_(key, num_probes) & capacity_minus_one);
			assert(num_probes < capacity_); // && "Hashtable is full: an error in key_equal<> or hash<>");
			//CHECK(num_probes < capacity_ && "Hashtable is full: an error in key_equal<> or hash<>") << " Key = " << key << ". Num of non-zero = " << nonzero_num();
		}
	}
}
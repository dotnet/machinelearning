#pragma once
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <cassert>
#include <map>
#include "data_block.h"
#include "hybrid_map.h"
#include "hybrid_alias_map.h"

namespace lda
{
	struct WordEntry
	{
		int32_t word_id_;
		int64_t offset_;
		int64_t end_offset_;
		int32_t capacity_;
		int32_t is_dense_;

		int32_t tf;
		int64_t alias_offset_;
		int64_t alias_end_offset_;
		int32_t alias_capacity_;
		int32_t is_alias_dense_;
	};

	class LDAModelBlock
	{
	public:
		LDAModelBlock();
		~LDAModelBlock();

		inline hybrid_map get_row(int word_id, int32_t *external_buf);
		inline hybrid_alias_map get_alias_row(int word_id);
		void SetWordInfo(int word_id, int32_t nonzero_num, bool fullSparse);

		void Clear();
		void Init(int32_t num_vocabs, int32_t num_topics);
		void Init(int32_t num_vocabs, int32_t num_topics, int64_t nonzero_num);
		void Init(int32_t num_vocabs, int32_t num_topics, int64_t mem_block_size, int64_t alias_mem_block_size);

		void InitFromDataBlock(const LDADataBlock *data_block, int32_t num_vocabs, int32_t num_topics);

		// void InitFromModelFile(const )
		//void SetNonzeroNum(int32_t word_id, int32_t nonzero_num);
		//void FinallizeNonzero();

		void GetModelStat(int64_t &mem_block_size, int64_t &alias_mem_block_size);

	private:

		LDAModelBlock(const LDAModelBlock &other) = delete;
		LDAModelBlock& operator=(const LDAModelBlock &other) = delete;

		void CountNonZero(std::vector<int32_t> &tfs);
		void InitModelBlockByTFS(bool fullSparse);
		void GetModelSizeByTFS(bool fullSparse, std::vector<int32_t> &tfs, int64_t &mem_block_size, int64_t &alias_mem_block_size);

		int32_t num_vocabs_;
		int32_t num_topics_;
		WordEntry *dict_;
		int32_t *mem_block_;
		int64_t mem_block_size_;

		int32_t *alias_mem_block_;
		int64_t alias_mem_block_size_;

		int64_t offset_;
		int64_t alias_offset_;

		const int32_t load_factor_ = 2;
		const int32_t sparse_factor_ = 5;
	};
	inline hybrid_map LDAModelBlock::get_row(int word_id, int32_t *external_buf)
	{
		hybrid_map row(mem_block_ + dict_[word_id].offset_,
			dict_[word_id].is_dense_,
			dict_[word_id].capacity_,
			0,
			external_buf);
		return row;
	}
	inline hybrid_alias_map LDAModelBlock::get_alias_row(int word_id)
	{
		hybrid_alias_map row(alias_mem_block_ + dict_[word_id].alias_offset_,
			dict_[word_id].is_alias_dense_,
			dict_[word_id].alias_capacity_);
		return row;
	}

}
// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once

#include <string>
#include <algorithm>
#include <memory>
#include "light_hash_map.h"

namespace lda
{
	class LDADocument;
	class LDADataBlock
	{
	public:
		explicit LDADataBlock(int32_t num_threads);
		~LDADataBlock();
		
		void Clear();
		//in data feedin scenario
		void Allocate(const int32_t num_document, const int64_t corpus_size);		
		//port the data from external process, e.g. c#
		int AddDense(int32_t* term_freq, int32_t term_num);
		int Add(int32_t* term_id, int32_t* term_freq, int32_t term_num);
		std::shared_ptr<LDADocument> GetOneDoc(int32_t index) const;

		inline int32_t num_documents() const;
		// Return the first document for thread thread_id
		inline int32_t Begin(int32_t thread_id) const;		
		// Return the next to last document for thread thread_i
		inline int32_t End(int32_t thread_id) const;


	private:
		LDADataBlock(const LDADataBlock& other) = delete;
		LDADataBlock& operator=(const LDADataBlock& other) = delete;

		int32_t num_threads_;
		bool has_read_;             // equal true if LDADataBlock holds memory

		int32_t index_document_;
		int64_t used_size_;

		int32_t num_documents_; 
		int64_t corpus_size_;

		int64_t* offset_buffer_;    // offset_buffer_ size = num_document_ + 1
		int32_t* documents_buffer_; // documents_buffer_ size = corpus_size_;
	};

	inline int32_t LDADataBlock::num_documents() const
	{
		return num_documents_;
	}
	inline int32_t LDADataBlock::Begin(int32_t thread_id) const
	{
		int32_t num_of_one_doc = num_documents_ / num_threads_;
		return thread_id * num_of_one_doc;
	}

	inline int32_t LDADataBlock::End(int32_t thread_id) const
	{
		if (thread_id == num_threads_ - 1)         // last thread
			return num_documents_;
		int32_t num_of_one_doc = num_documents_ / num_threads_;
		return (thread_id + 1) * num_of_one_doc;
	}
}
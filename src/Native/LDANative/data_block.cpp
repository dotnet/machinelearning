
#include <iostream>
#include "data_block.h"
#include "lda_document.h"

namespace lda
{
	LDADataBlock::LDADataBlock(int32_t num_threads) : 
		num_threads_(num_threads), has_read_(false), index_document_(0), documents_buffer_(nullptr), offset_buffer_(nullptr)
	{
	}

	LDADataBlock::~LDADataBlock() 
	{
		if (has_read_)
		{
			delete[] offset_buffer_;
			delete[] documents_buffer_;
		}
	}

	void LDADataBlock::Clear()
	{
		has_read_ = false;
		index_document_ = 0;
		used_size_ = 0;

		num_documents_ = 0;
		corpus_size_ = 0;

		if (offset_buffer_)
		{
			delete[]offset_buffer_;
			offset_buffer_ = nullptr;
		}
		if (documents_buffer_)
		{
			delete[]documents_buffer_;
			documents_buffer_ = nullptr;
		}
	}

	void LDADataBlock::Allocate(const int32_t num_document, const int64_t corpus_size)
	{
		num_documents_ = num_document;
		corpus_size_ = corpus_size;

		offset_buffer_ = new int64_t[num_documents_ + 1]; // +1: one for the end of last document,
		documents_buffer_ = new int32_t[corpus_size_];

		index_document_ = 0;
		used_size_ = 0;

		offset_buffer_[0] = 0;
	}


	//term_id, term_freq, term_num
	int LDADataBlock::Add(int32_t* term_id, int32_t* term_freq, int32_t term_num)
	{
		int64_t data_length = 1;
		// offset_buffer_[index_document_] = used_size_;

		int64_t idx = offset_buffer_[index_document_] + 1;
		for (int i = 0; i < term_num; ++i)
		{
			for (int j = 0; j < term_freq[i]; ++j)
			{
				documents_buffer_[idx++] = term_id[i];
				documents_buffer_[idx++] = 0;
				data_length += 2;
			}
		}

		index_document_++;
		used_size_ += data_length;

		offset_buffer_[index_document_] = used_size_;
		has_read_ = true;

		return (int)data_length;
	}

	int LDADataBlock::AddDense(int32_t* term_freq, int32_t term_num)
	{
		int64_t data_length = 1;
		// offset_buffer_[index_document_] = used_size_;

		int64_t idx = offset_buffer_[index_document_] + 1;
		for (int i = 0; i < term_num; ++i)
		{
			for (int j = 0; j < term_freq[i]; ++j)
			{
				documents_buffer_[idx++] = i;
				documents_buffer_[idx++] = 0;
				data_length += 2;
			}
		}

		index_document_++;
		used_size_ += data_length;

		offset_buffer_[index_document_] = used_size_;
		has_read_ = true;

		return (int)data_length;
	}

	std::shared_ptr<LDADocument> LDADataBlock::GetOneDoc(int32_t index) const
	{
		std::shared_ptr<LDADocument> returned_ptr(
			new LDADocument(documents_buffer_ + offset_buffer_[index],
			documents_buffer_ + offset_buffer_[index + 1]
			)
			);
		return returned_ptr;
	}
}
#ifndef _SIMPLE_BARRIER_H_
#define _SIMPLE_BARRIER_H_

#include <atomic>
#include <condition_variable>
#include <thread>
namespace lda
{
	class SimpleBarrier
	{
	public:
		SimpleBarrier(unsigned int n) :barrier_size_(n), num_of_waiting_(0), rounds_(0)
		{};

		void reset()
		{
			throw "not implemented yet.";
		}

		bool wait()
		{
			std::unique_lock<std::mutex> lock(mutex_);
			if (num_of_waiting_.fetch_add(1) >= barrier_size_ - 1)
			{
				cond_.notify_all();
				num_of_waiting_.store(0);
				rounds_.fetch_add(1);
				return true;
			}
			else
			{
	
				unsigned int i = rounds_.load();
				cond_.wait(lock, [&]{return i != rounds_.load(); });
				return false;
			}
		}

		~SimpleBarrier()
		{
			num_of_waiting_ = 0;
			rounds_ = 0;
		}



	protected:
		const unsigned int barrier_size_;

		std::atomic<unsigned int> num_of_waiting_;
		std::atomic<unsigned int> rounds_;
		std::condition_variable cond_;
		std::mutex mutex_;
	};
}





#endif //  _SIMPLE_BARRIER_H_


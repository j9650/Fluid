#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <cassert>
#include <thread>
#include <pthread.h>
#include <mutex>
#include "../apps/kmeans.h"
#include "../utils/logger.h"
namespace sth {
	static const int kStartThread = 80;
	static const int kSuspendThread = 81;
	static const int kResumeThread = 82;
	static const int kTerminateThread = 83;

};

class StoppableThread {
protected:
	int status;
	int suspendcount;
	std::string name;
	std::vector<StoppableThread* > parents;
	std::vector<StoppableThread* > children;
public:

	static std::mutex mtx;
	static std::random_device rd;
	static bool rdlk;
	static uint32_t random() {
		uint32_t at = 0;
		if (rdlk) {
			std::lock_guard<std::mutex> lk(mtx);
			at = rd();
		}
		else {
			at = rd();
		}

		return at;
	}

	std::thread thread;
	std::thread::native_handle_type threadhandle;

	StoppableThread() {
		suspendcount  = 0;
	};
	inline int getState() const {
		return status;
	}

	template<typename FunctionType, typename ... Args>
	void restart(FunctionType fun, Args ... args) {
		//TerminateThread(threadhandle, 0);
		thread = std::thread(fun, args...);
		//123//pthread_create(&thread, NULL, fun, args...);
		threadhandle = thread.native_handle();
	}


	void detach() {
		thread.detach();
	}

	void suspend() {
		assert(0);
	}
	
	void suspend(StoppableThread* t) {
		t->suspend();
	}

	void resume() {
		//ResumeThread(threadhandle);
	}

	void terminate() {
		{
			//assert(0);
			if (rdlk == false) {
				rdlk = true;
				random();
			}
			std::lock_guard<std::mutex> lk(mtx);

			unsigned long long th = threadhandle;
			pthread_cancel(pthread_t(threadhandle));
			SyncLogger::print("Cancel Thread: ", name, " Canceling", " (", std::to_string(th), ")");
			rdlk = false;
		}
	}

	void join() {
		thread.join();
	}
};


class DetachedThread : public StoppableThread {
private:
public:
	template<typename FunctionType, typename ... Args>
	DetachedThread(std::string threadname, FunctionType fun, Args ... args)  {
		name = threadname;
		suspendcount = -1;

		thread = std::thread(fun, args...);
		threadhandle = thread.native_handle();
		unsigned long long th = threadhandle;
		SyncLogger::print("Detached Thread: ", name, " Running", " (", std::to_string(th), ")");
		detach();
		status = sth::kStartThread;
	}
};
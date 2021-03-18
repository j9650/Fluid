#pragma once
#include "../fluid/guard.h"
#include "../fluid/valve.h"
#include "../fluid/fluid.h"
#include "../fluid/guardscheduler.h"
#include "laplacian.h"
namespace example {

	
	class LaplacianFluid_E : public Laplacian {
	public:
		////#pragma valve{ValveGT<int> v1}
		ValveGT<int> v1;
		std::vector<Guard*> guard_log;
		LaplacianFluid_E() {
					
		}
		virtual void segmentImage(tifImage *image);

		////virtual void Gaussian_filter(tifImage *image);
		virtual void Gaussian_filter(tifImage *image, int *call_num);
		virtual void Mean_filter(tifImage *image, int *call_num);
		////virtual void Sobel_filter(tifImage *image);
		virtual void Sobel_filter(tifImage *image);
		virtual void Laplacian_filter(tifImage *image);
	};

	class LaplacianFluid_multi_E : public Laplacian {
	public:
		////#pragma valve{ValveGT<int> v1}
		ValveGT<int> v1;
		ValveGT_vec<int> v2;
		int thread_num;
		LaplacianFluid_multi_E(int thread_num_) {
			thread_num = thread_num_;
		}
		virtual void segmentImage(tifImage *image);

		////virtual void Gaussian_filter(tifImage *image);
		virtual void Gaussian_filter(tifImage *image, int num_thread, std::vector<int> *call_num);
		// virtual void Gaussian_filter_threads(LaplacianFluid_multi_E *sb, tifImage *image, double *kernel, int num_thread, int thread_id, int *call_num);
		virtual void Gaussian_filter_threads(LaplacianFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id, int *call_num);
		virtual void Gaussian_filter_threads2(LaplacianFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id, int *call_num);

		////virtual void Sobel_filter(tifImage *image);
		virtual void Sobel_filter(tifImage *image, int num_thread);
		// virtual void Sobel_filter_threads(LaplacianFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id);
		virtual void Sobel_filter_threads(LaplacianFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id);
		virtual void Sobel_filter_threads2(LaplacianFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id);
	};
};
#pragma once
#include "../fluid/guard.h"
#include "../fluid/valve.h"
#include "../fluid/fluid.h"
#include "../fluid/guardscheduler.h"
#include "sobel.h"
namespace example {

	//class SobelFluid : public SobelFluid {
	//public:
	//	//virtual void Coloring(GraphGC *graph);
	//};


	////__Fluid__
	////class SobelFluid : public Sobel {
	class SobelFluid : public Sobel {
	public:
		////#pragma valve{ValveGT<int> v1}
		ValveGT<int> v1;
		SobelFluid() {
					
		}
		virtual void segmentImage(tifImage *image);

		////virtual void Gaussian_filter(tifImage *image);
		virtual void Gaussian_filter(tifImage *image, int *call_num);
		////virtual void Sobel_filter(tifImage *image);
		virtual void Sobel_filter(tifImage *image);
		virtual void Link_edge(tifImage *image);
		virtual void Segment(tifImage *image);
	};
	
	class SobelFluid_E : public Sobel {
	public:
		////#pragma valve{ValveGT<int> v1}
		ValveGT<int> v1;
		std::vector<Guard*> guard_log;
		SobelFluid_E() {
					
		}
		virtual void segmentImage(tifImage *image);

		////virtual void Gaussian_filter(tifImage *image);
		virtual void Gaussian_filter(tifImage *image, int *call_num);
		////virtual void Sobel_filter(tifImage *image);
		virtual void Sobel_filter(tifImage *image);
	};

	class SobelFluid_multi_E : public Sobel {
	public:
		////#pragma valve{ValveGT<int> v1}
		ValveGT<int> v1;
		ValveGT_vec<int> v2;
		int thread_num;
		SobelFluid_multi_E(int thread_num_) {
			thread_num = thread_num_;
		}
		virtual void segmentImage(tifImage *image);

		////virtual void Gaussian_filter(tifImage *image);
		virtual void Gaussian_filter(tifImage *image, int num_thread, std::vector<int> *call_num);
		// virtual void Gaussian_filter_threads(SobelFluid_multi_E *sb, tifImage *image, double *kernel, int num_thread, int thread_id, int *call_num);
		virtual void Gaussian_filter_threads(SobelFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id, int *call_num);
		virtual void Gaussian_filter_threads2(SobelFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id, int *call_num);

		////virtual void Sobel_filter(tifImage *image);
		virtual void Sobel_filter(tifImage *image, int num_thread);
		// virtual void Sobel_filter_threads(SobelFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id);
		virtual void Sobel_filter_threads(SobelFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id);
		virtual void Sobel_filter_threads2(SobelFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id);
	};
};
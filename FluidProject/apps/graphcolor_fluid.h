#pragma once
#include "../fluid/guard.h"
#include "../fluid/valve.h"
#include "../fluid/fluid.h"
#include "../fluid/guardscheduler.h"
#include "graphcolor.h"
namespace example {

	class GraphcolorFluidX : public GraphcolorFluid {
	public:
		ValveGT<int> v1;
		virtual void Coloring(GraphGC *graph);
	};

	class GraphcolorFluidX_E : public GraphcolorFluid {
	public:
		ValveGT<int> v1;
		std::vector<Guard*> guard_log;
		virtual void Coloring(GraphGC *graph);
		virtual void Kernel(GraphGC *graph, int iter, int *call_num);
		virtual void Docolor(GraphGC *graph, int iter, int *tt, int *call_num);
	};
	
};
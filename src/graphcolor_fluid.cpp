#include "graphcolor.h"
#include "graphcolor_fluid.h"

using namespace example;

/*
void GraphcolorFluidX::ShortestPath(GraphGC *graph) {

	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(2);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

	int* call_num_ = new int(0);
	auto tp1status_ = new __fluid__<int>(call_num_); //__call_num__  new call_num_;
	auto tpb1 = std::bind(&GraphcolorFluid::Relaxing_edge, this, graph, std::placeholders::_1);
	auto tp1 = tf->newTask<decltype(tpb1), int*>("Relaxing_edge" + std::to_string(0), tpb1, call_num_);
	Guard* g0 = gs->newGuard("GuardRelaxing_edge" + std::to_string(0), { v0 }, tp1, {});

	int* call_num = new int(0);
	auto tp1status = new __fluid__<int>(call_num);
	for (int iter = 1; iter < graph->v_num; iter++) {

		call_num = call_num_;
		tp1status = tp1status_; // __call_num__ call_num = call_num_;

		call_num_ = new int(0);
		tp1status_ = new __fluid__<int>(call_num_); //__call_num__  call_num_.clear();

		auto v0 = new ValveGT<int>(tp1status, graph->e_num * rate); //ValveGT(call_num,  graph->e_num * 0.5);
		auto tpb1 = std::bind(&GraphcolorFluid::Relaxing_edge, this, graph, std::placeholders::_1);
		auto tp1 = tf->newTask<decltype(tpb1), int*>("Relaxing_edge" + std::to_string(iter), tpb1, call_num_);
		Guard* g1 = gs->newGuard("GuardRelaxing_edge" + std::to_string(iter), { v0 }, tp1, {g0});

		g0 = g1;

		//Relaxing_edge(graph,call_num);
		//std::cout << "iter: " << iter <<std::endl;
	}
	auto tp3 = tf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	auto v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	gs->sync(tp3);
	return;
} */

void GraphcolorFluidX::Coloring(GraphGC *graph)
{
	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(2);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

  	int tt = graph->v_num;
  	int iter = 0;
  	double rate = this->rate;
	//auto v1 = new ValveGT<int>(call_num, rate*tt);


	while(tt) {
    	iter++;

    	//#pragma call_num {__call_num__ call_num;}
    	int* anything = new int(0);
		auto call_num = new __fluid__<int>(anything);

		//<<<g1,{},{},call_num>>>Kernel(graph, iter);
		auto tpb__g1 = std::bind(&GraphcolorFluid::Kernel, this, graph, iter, std::placeholders::_1);
		auto tp__g1 = tf->newTask<decltype(tpb__g1), int*>("Kernel" + std::to_string(iter), tpb__g1, call_num->p);
		Guard* g1 = gs->newGuard("GuardKernel" + std::to_string(iter), {}, tp__g1, {});

		if(tt < (graph->v_num/50)) rate = 1;
		//<<<g2,{v1(call_num,rate*tt)},{g1}>>>Docolor(graph, iter, &tt);
		auto v0 = v1.init(call_num, rate*graph->v_num);
		auto tpb__g2 = std::bind(&Graphcolor::Docolor, this, graph, iter, &tt);
		auto tp__g2 = tf->newTask<decltype(tpb__g2)>("Docolor" + std::to_string(iter), tpb__g2);
		Guard*       g2 = gs->newGuard("Docolor" + std::to_string(iter), { v0 }, tp__g2, {  });

		gs->synctask(tp__g2);
		//gs->sync(tp__g2);

    	//Kernel(graph, iter);
    	//Docolor(graph, iter, &tt);
    	std::cout << "tt=" << tt<<std::endl;
  }
  gs->sync(tp0);
  std::cout << "num of color: " << iter << std::endl;
}

void GraphcolorFluidX_E::Coloring(GraphGC *graph)
{
	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(2);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

  	int tt = graph->v_num;
  	int iter = 0;
  	double rate = this->rate;
	//auto v1 = new ValveGT<int>(call_num, rate*tt);


	while(tt) {
    	iter++;

		Data *d1 = new Data;
		d1->set_data((void*)graph);
		Data *d2 = new Data;
		d2->set_data((void*)graph);
		Data *d3 = new Data;
		d3->set_data((void*)graph);

    	//#pragma call_num {__call_num__ call_num;}
    	int* anything = new int(0);
		auto call_num = new __fluid__<int>(anything);

		//<<<g1,{},{},call_num>>>Kernel(graph, iter);
		auto tpb__g1 = std::bind(&GraphcolorFluidX_E::Kernel, this, graph, iter, std::placeholders::_1);
		auto tp__g1 = tf->newTask<decltype(tpb__g1), int*>("Kernel" + std::to_string(iter), {d1}, {d2}, tpb__g1, call_num->p);
		Guard* g1 = gs->newGuard("GuardKernel" + std::to_string(iter), {}, {}, tp__g1, {});
		g1->set_root();
  		guard_log.push_back(g1);

    	//#pragma call_num {__call_num__ call_num2;}
    	int* call_num2 = new int(0);
		auto anything_call_num2 = new __fluid__<int>(call_num2);

		if(tt < (graph->v_num/50)) rate = 1;
		//<<<g2,{v1(call_num,rate*tt)},{g1}>>>Docolor(graph, iter, &tt);
		int tt_;
		auto v0 = v1.init(call_num, rate*graph->v_num);
		auto v1_ = v1.init(anything_call_num2, end_quality);
		auto tpb__g2 = std::bind(&GraphcolorFluidX_E::Docolor, this, graph, iter, &tt, call_num2);
		auto tp__g2 = tf->newTask<decltype(tpb__g2)>("Docolor" + std::to_string(iter), {d2}, {d3}, tpb__g2);
		Guard*       g2 = gs->newGuard("Docolor" + std::to_string(iter), { v0 }, {v1_}, tp__g2, {  });
		g2->set_leaf();
  		guard_log.push_back(g2);

		gs->synctask(tp__g2);
		//gs->sync(tp__g2);

    	//Kernel(graph, iter);
    	//Docolor(graph, iter, &tt);
    	std::cout << "tt=" << tt<<std::endl;
  }
  gs->sync(tp0);
  std::cout << "num of color: " << iter << std::endl;
}

void GraphcolorFluidX_E::Docolor(GraphGC *graph, int iter, int *tt, int *call_num)
{
  //int tot = 0;
  //for(int t=0; t<10; t++)
  for(int i=0; i<graph->v_num; i++) {
    if(graph->ga[i] && graph->color[i].color == 0)
    {
      //if(i == 3999 || i == 466) std::cout <<  i << ":" << iter << ":" << graph->ga[i] << std::endl;
      graph->color[i].color = graph->ga[i];
      (*tt)--;
    }
    if (graph->ga[i] == iter) (*call_num)++;
  }
  //(*tt) += (*call_num);
  std::cout << "Docolor wan le!\n";
}

void GraphcolorFluidX_E::Kernel(GraphGC *graph, int iter, int *call_num)
{
  if(iter%2 != 100)
  {
    for(int i=0; i<graph->v_num; i++) {
      //if(graph->color[i].color == 0)
      huafen(graph, i, iter),
      (*call_num)++;
      //print_to_logs(graph,"call_num: ");
      //std::cout << "call_num: " << (*call_num) << std::endl;
    }
  }
  else
  {
    for(int i=graph->v_num-1; i>=0; i--) {
      //if(graph->color[i].color == 0)
      huafen(graph, i, iter),
      (*call_num)++;
      //print_to_logs(graph,"call_num: ");
      //std::cout << "call_num: " << (*call_num) << std::endl;
    }
  }
  //std::cout << "Kernel wan le!\n";
}
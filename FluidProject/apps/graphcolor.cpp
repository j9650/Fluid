#include "graphcolor.h"
#include "../utils/logger.h"
#define _CRT_RAND_S
#include "stdlib.h"
#include "../fluid/thread.h"
#include <cstdlib>
#include <cstdio>
#include <climits>
#include <ctime>
#include <random>
#include <chrono>
#include <sstream>      // std::stringstream, std::stringbuf
#include <unordered_set>
#include <set>
#include <vector>

using namespace example;	

void print_to_logs(GraphGC *graph, std::string st)
{
  graph->logs.push_back(st);
}

int Graphcolor::graphcolor(std::string in) {

	GraphGC graph;
	InitGraph(&graph,in);

  Randomcolor(&graph);
	Coloring(&graph);

  Checkcolor(&graph);
	printf("Graphcolor!\n");
	return 0;
}

void Graphcolor::Randomcolor(GraphGC *graph)
{
  graph->randlist = new int[graph->v_num];
  graph->color = new COLOR[graph->v_num];
  graph->ga = new int[graph->v_num];
  graph->shn = new int[graph->v_num];
  for(int i=0; i<graph->v_num; i++) {
    graph->randlist[i] = rand() % graph->v_num;
    graph->color[i].color = 0;
    graph->ga[i] = 0;
    graph->shn[i] = 0;
  }
}

void Graphcolor::Checkcolor(GraphGC *graph)
{
  int x,y;
  x = 0;
  for(int i=0; i<graph->e_num; i++)
  {
    while(graph->vertex[x+1] <= i) x++;
    y = graph->edge[i];
    if(graph->color[x].color == graph->color[y].color)
      std::cout << "error! color[" << x << "] == color[" << y << "] == " << graph->color[x].color << std::endl;
  }
  std::cout << "result correct!\n";

}

void Graphcolor::InitGraph(GraphGC *graph, std::string in)
{
	//std::ifstream file (in);
  	std::ifstream file;
  	file.open(in, std::ifstream::in);

    std::cout << "Read data from: " << in << std::endl;

  	file >> graph->v_num;
  	file >> graph->e_num;
    std::cout << "v_num: " << graph->v_num << " e_num: " << graph->e_num << std::endl;
  	graph->e_num = graph->e_num * 2;
    std::cout << "OK" << std::endl;
  	graph->vertex = new int[graph->v_num+1];
    std::cout << "OK" << std::endl;
  	graph->edge = new int[graph->e_num ];
    std::cout << "OK" << std::endl;
  	graph->dist = new COLOR_[graph->v_num];

  	std::stringstream ss;
  	std::string st;

  	int tot=0;
  	graph->vertex[0] = 0;
  	std::getline(file, st);
  	for (int i=0; i< graph->v_num; i++)
  	{
  		std::getline(file, st);
  		ss.clear();
  		ss.str("");
  		ss << st;
  		int e;
  		while(ss >> e)
  		{
  			graph->edge[tot++] = e;
  		}
  		graph->vertex[i+1] = tot;
  	}
  	std::cout << "tot: " << tot << std::endl;

    graph->weights = new int[graph->e_num];
    for(int i=0; i<graph->e_num; i++)
      graph->weights[i] = graph->edge[i] % 97;
    for(int i=0; i<graph->v_num; i++)
      graph->dist[i].dist = 55555555;
    for(int i=0; i<graph->vertex[1]; i++)
      graph->dist[graph->edge[i]].dist = graph->weights[i];
    graph->dist[0].dist = 0;
}

void Graphcolor::Coloring(GraphGC *graph)
{
  int tt = graph->v_num;
  int iter = 0;

  while(tt) {
    iter++;
    Kernel(graph, iter);
    Docolor(graph, iter, &tt);
    //std::cout <<tt<<std::endl;
  }
  std::cout << "num of color: " << iter << std::endl;
}

void Graphcolor::huafen(GraphGC *graph, int vid, int iter)
{
  int found_larger=0;
  int local_rand = graph->randlist[vid];
  for(int i=graph->vertex[vid]; i<graph->vertex[vid+1]; i++) {
    int dest = graph->edge[i];
    if((graph->color[dest].color && graph->ga[dest] < iter)||(graph->ga[dest] < iter && graph->ga[dest])) continue;
    if((graph->randlist[dest]>local_rand) || (graph->randlist[dest]==local_rand && dest<vid)) found_larger = 1; 
  }
  if(!found_larger)
  {
    if(graph->ga[vid] == 0)  graph->ga[vid] = iter;
  }
}

void Graphcolor::Kernel(GraphGC *graph, int iter)
{ 
  if(iter%2 != 0)
  {
    for(int i=0; i<graph->v_num; i++) {
      huafen(graph, i, iter);
    }
  }
  else
  {
    for(int i=graph->v_num-1; i>=0; i--) {
      huafen(graph, i, iter);
    }
  }
}

void Graphcolor::Docolor(GraphGC *graph, int iter, int *tt)
{
  for(int i=0; i<graph->v_num; i++) {
    if(graph->ga[i] && graph->color[i].color == 0)
    {
      graph->color[i].color = graph->ga[i];
      (*tt)--;
    }
  }
  //std::cout << "Docolor wan le!\n";
}

void GraphcolorFluid::Kernel(GraphGC *graph, int iter, int *call_num)
{
  if(iter%2 != 100)
  {
    for(int i=0; i<graph->v_num; i++) {
      huafen(graph, i, iter),
      (*call_num)++;
    }
  }
  else
  {
    for(int i=graph->v_num-1; i>=0; i--) {
      huafen(graph, i, iter),
      (*call_num)++;
    }
  }
}


//////////////////////////////////////////multi-thread///////////////////////////////////////////////////////

void Kernel_thread(Graphcolor_multi *gc, GraphGC *graph, int iter, int thread_num, int id) {
  if(iter%2 != 100)
  {
    for(int i=id; i<graph->v_num; i+=thread_num) {
      gc->huafen(graph, i, iter);
    }
  }
  else
  {
    for(int i=graph->v_num-1-id; i>=0; i-=thread_num) {
      gc->huafen(graph, i, iter);
    }
  }
}

void Graphcolor_multi::Kernel(GraphGC *graph, int iter)
{
  std::thread* threads = new std::thread[this->thread_num];
  for (int id = 0; id < this->thread_num; id++) {
    threads[id] = std::thread(Kernel_thread, this, graph, iter, this->thread_num, id);
  }
  //Kernel_thread(this, graph, iter, this->thread_num, 0);
  for(int i = 0; i < this->thread_num; i++) {
    threads[i].join();
  }
}


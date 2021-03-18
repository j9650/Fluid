#pragma once
#include <cmath>
#include <map>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <random>
#include <mutex>
#include <cassert>
#include <vector>
#include <stdio.h>
#include <iostream>

//http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.302.9543&rep=rep1&type=pdf

#define MAX_COUNT 1200000
#define STABLE_TH 6
#define STABLE_DI 1
namespace example {

	class Medusa {
	public:
		//static std::string loopindicator;
		double rate;
		double sum;
		int dups;
		double sigma;
		double iou;

		int iter;
		int select_num;
		int rmsd_th;

		std::string input_file_name;
		int num_images;
		int input_height;
		int input_width;
		int input_channel;
		std::vector<double> rmsd;
		std::vector<double> energy;
		int success_protein;

		double accuracy;

		virtual void medusa(std::string input_path, std::string in);
		virtual void medusa_region(std::vector<std::string>* filename_vec);
		virtual int check_result();
		virtual void medusa_dock(std::vector<double>* energy, std::vector<double>* rmsd, std::vector<std::string>* filename_vec);
		virtual void select(std::vector<double>* energy, std::vector<double>* rmsd);

		Medusa(int iter_, int select_num_, int rmsd_th_) : iter(iter_), select_num(select_num_), rmsd_th(rmsd_th_)
		{  
		}

	};
	
}
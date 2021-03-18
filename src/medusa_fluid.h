#pragma once
#include "../fluid/guard.h"
#include "../fluid/valve.h"
#include "../fluid/fluid.h"
#include "../fluid/guardscheduler.h"
#include "medusa.h"
namespace example {
	
	class MedusaFluid_E : public Medusa {
	public:
		////#pragma valve{ValveGT<int> v1}
		ValveGT<int> v1;
		std::vector<Guard*> guard_log;
		MedusaFluid_E(int iter_, int select_num_, int rmsd_th_) : Medusa(iter_, select_num_, rmsd_th_)//iter(iter_), select_num(select_num_), rmsd_th(rmsd_th_)
		{  
		}

		virtual void medusa_region(std::vector<std::string>* filename_vec);
		virtual void medusa_dock(std::vector<double>* energy, std::vector<double>* rmsd, std::vector<std::string>* filename_vec, int *cc, int *tot);
		virtual void select(std::vector<double>* energy, std::vector<double>* rmsd);
	};
};
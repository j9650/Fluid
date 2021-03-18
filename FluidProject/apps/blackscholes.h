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

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <fstream>
#include <iomanip>

#define DIVIDE 120.0


//Precision to use for calculations
#define fptype float

#define NUM_RUNS 100

namespace example {

typedef struct OptionData_ {
        fptype s;          // spot price
        fptype strike;     // strike price
        fptype r;          // risk-free interest rate
        fptype divq;       // dividend rate
        fptype v;          // volatility
        fptype t;          // time to maturity or option expiration in years 
                           //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)  
        char OptionType;   // Option type.  "P"=PUT, "C"=CALL
        fptype divs;       // dividend vals (not used in this test)
        fptype DGrefval;   // DerivaGem Reference Value
} OptionData;



fptype CNDF ( fptype InputX );
fptype BlkSchlsEqEuroNoDiv( fptype sptprice,
                            fptype strike, fptype rate, fptype volatility,
                            fptype time, int otype, float timet, fptype* N1, fptype* N2);
double normalize(double in, double min, double max, double min_new, double max_new);


	class Blackscholes {
	public:
		//static std::string loopindicator;
		double fluid_rate;
		double sum;
		int dups;
		double sigma;
		double iou;

		int select_num;
		OptionData *data;
		fptype *prices;
		int numOptions;
		
		int    * otype;
		fptype * sptprice;
		fptype * strike;
		fptype * rate;
		fptype * volatility;
		fptype * otime;
		int numError = 0;

		std::string input_file_name;

		double accuracy;

		virtual void blackscholes(std::string input_path, std::string in);
		virtual void region();
		virtual int bs_thread(void *tid_ptr);

		Blackscholes(int select_num_) : select_num(select_num_)
		{  
		}

	};

}  // namespace example
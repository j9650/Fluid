#include "medusa.h"
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
#include <stdio.h>
#include <math.h> 

using namespace example;

void Medusa::medusa(std::string input_path, std::string in) {
  std::cout << "Medusa::medusa\n";
  std::ifstream input_file;
  input_file.open(input_path + in, std::ifstream::in);

  int protein_num;
  int iter_per_protein;
  input_file >> protein_num;
  input_file >> iter_per_protein;

  std::vector<std::string> filename_vec;
  //std::vector<std::vector<double>> label;
  filename_vec.resize(iter_per_protein);
  //label.resize(iter_per_protein);
  //std::vector<double> label_vec;
  int size;
  double label;

  success_protein = 0;
  for (int i = 0; i < protein_num; i++) {
    rmsd.resize(0);
    energy.resize(0);
 	  for (int j = 0; j < iter_per_protein; j++){
      input_file >> filename_vec[j];
 	  	input_file >> size;
 	  	//label_vec.resize(size);
  
 	  	for (int s = 0; s < size; s++){
 	  		input_file >> label;
        rmsd.push_back(label);
        //std::cout << label << std::endl;
 	  	}
 	  	//label[j] = label_vec;
 	  }

  	medusa_region(&filename_vec);

  	success_protein += check_result();
  }

  std::cout << "success_proteins: " << success_protein << std::endl;
  accuracy = success_protein;
}

void Medusa::medusa_region(std::vector<std::string>* filename_vec) {
  //std::vector<int> energys;
  //std::vector<int> labels;
  
  medusa_dock(&energy, &rmsd, filename_vec);
  select(&energy, &rmsd);
}

void Medusa::medusa_dock(std::vector<double>* energy, std::vector<double>* rmsd, std::vector<std::string>* filename_vec) {
  std::cout << "Medusa::medusa_dock" << std::endl;
  std::ifstream pdb_file;
  std::string st;
  double this_energy;
  double min_energy = 0.0;
  int min_iter = 0;
  int cc = 0;
  std::vector<std::string> file;
  for (int i = 0; i < filename_vec->size(); i++) {
    pdb_file.open((*filename_vec)[i], std::ifstream::in);
    while(std::getline(pdb_file, st)) {  //changed
      file.push_back(st);
    }
    pdb_file.close();
    //std::cout << file.size() << std::endl;
    //for (int iiiii = 0; iiiii < file.size(); iiiii++) {
    for (auto st : file) {
      //st = file[iiiii];
      if (st.find("E_total:") != std::string::npos) {
        //std::cout << st << std::endl;
        //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        this_energy = std::stod(st.substr(st.find(":") + 2));
        energy->push_back(this_energy);
        if (this_energy < min_energy) {
          min_energy = this_energy;
          min_iter = i;
          cc = 0;
        }
      }
    }
    cc++;
    file.resize(0);
    if (cc >= rate) {
      //std::cout << cc << ' ' << i << std::endl;
      continue;
    } else {
      //std::cout << cc << ' ' << i << std::endl;
    }
  }
}

void Medusa::select(std::vector<double>* energy, std::vector<double>* rmsd) {
  std::cout << "Medusa::select" << std::endl;
  double tmp;
  std::cout << energy->size() << ' ' << rmsd->size() << std::endl;
  for (int i = 0; i < select_num; i++) {
    for (int j = i + 1; j < energy->size(); j++) {
      if ((*energy)[i] > (*energy)[j]) {
        tmp = (*energy)[i];
        (*energy)[i] = (*energy)[j];
        (*energy)[j] = tmp;
        tmp = (*rmsd)[i];
        (*rmsd)[i] = (*rmsd)[j];
        (*rmsd)[j] = tmp;
      }
    }
  }
}

int Medusa::check_result() {
  std::cout << "Medusa::check_result" << std::endl;
  int tot = 0;
  for (int i = 0; i < select_num; i++) {
    if (rmsd[i] < rmsd_th) {
      return 1;
    }
  }

  return 0;
}
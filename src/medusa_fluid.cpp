#include "medusa.h"
#include "medusa_fluid.h"

using namespace example;

void MedusaFluid_E::medusa_region(std::vector<std::string>* filename_vec) {
  std::cout << "MedusaFluid_E::medusa_region" << std::endl;
  auto tf = new TaskFactory();
  auto gs = new AggressiveGS(20);
  auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
  auto tp0status = new __fluid__<bool>(tp0->isfinished());
  auto v0 = new ValveEQ<bool>(tp0status, true);
  tp0->run();

  Data *d1 = new Data;
  d1->set_data((void*)(&energy));
  Data *d2 = new Data;
  d2->set_data((void*)(&energy));
  Data *d3 = new Data;
  d3->set_data((void*)(&energy));


  ////#pragma call_num call_num1;
  int *call_num1__anything = new int(0);
  auto call_num1 = new __fluid__<int>(call_num1__anything);

  ////#pragma call_num call_num2;
  int *call_num2__anything = new int(0);
  auto call_num2 = new __fluid__<int>(call_num2__anything);

  ////<<<g1,{},{},{d1},{d2}>>>medusa_dock(&energy, &rmsd, filename_vec);
  auto tpb__g1 = std::bind(&MedusaFluid_E::medusa_dock, this, &energy, &rmsd, filename_vec, call_num1->p, call_num2->p);
  auto tp__g1 = tf->newTask<decltype(tpb__g1)>("medusa_dock0", {d1}, {d2}, tpb__g1);
  Guard* g1 = gs->newGuard("medusa_dock0", {}, {}, tp__g1, {});
  ////g1->set_root;
  g1->set_root();
  guard_log.push_back(g1);


  ////#pragma valve v0_ = v1.init(call_num1, rate);
  //rate = rate * filename_vec->size() / 100.0;
  std::cout << "threshold: " << filename_vec->size() * rate << std::endl;
  auto v0_ = v1.init(call_num1, filename_vec->size() * rate);
  auto v1_ = v1.init(call_num2, iter);
  ////<<<g2,{v0_},{v1_},{d2},{d3}>>select(&energy, &rmsd);
  auto tpb__g2 = std::bind(&MedusaFluid_E::select, this, &energy, &rmsd);
  auto tp__g2 = tf->newTask<decltype(tpb__g2)>("select0", {d2}, {d3}, tpb__g2);
  Guard* g2 = gs->newGuard("select0", {v0_}, {v1_}, tp__g2, {g1});
  /////g2->set_leaf();
  g2->set_leaf();
  guard_log.push_back(g2);
  gs->synctask(tp__g2);
  std::cout << "tot:                  " << *(call_num2->p) << std::endl;
  //gs->sync(tp__g2);
}

void MedusaFluid_E::medusa_dock(std::vector<double>* energy, std::vector<double>* rmsd, std::vector<std::string>* filename_vec, int *cc, int *tot) {
  std::cout << "MedusaFluid_E::medusa_dock Start!" << std::endl;
  std::ifstream pdb_file;
  std::string st;
  double this_energy;
  double min_energy = 0.0;
  int min_iter = 0;
  *cc = 0;
  int ccc = 0;
  std::vector<std::string> file;
  for ((*tot) = 0; (*tot) < filename_vec->size(); (*tot)++) {
    pdb_file.open((*filename_vec)[(*tot)], std::ifstream::in);
    while(std::getline(pdb_file, st)) {  //changed
      file.push_back(st);
    }
    pdb_file.close();
    for (auto st : file) {
      if (st.find("E_total:") != std::string::npos) {
        //std::cout << st << std::endl;
        //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        this_energy = std::stod(st.substr(st.find(":") + 2));
        energy->push_back(this_energy);
        if (this_energy < min_energy) {
          min_energy = this_energy;
          min_iter = *tot;
          ccc = 0;
          //*cc = 0;
        }
      }
    }
    //(*cc)++;
    ccc++;
    file.resize(0);
    // if (cc > th) {
    //  break;
    // }
    //std::cout << ccc << ' ' << *tot << std::endl;
    if (ccc >= rate * filename_vec->size()) {
    	*cc = rate * filename_vec->size() + 1;
    }// else {
    //	//std::cout << ccc << ' ' << *tot << std::endl;
    //}
    //(*cc)++;
  }
  (*cc) = rate * filename_vec->size() + 1;
  std::cout << "MedusaFluid_E::medusa_dock End!" << std::endl;
}

void MedusaFluid_E::select(std::vector<double>* energy, std::vector<double>* rmsd) {
  std::cout << "MedusaFluid_E::select Start!" << std::endl;
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
  std::cout << "MedusaFluid_E::select End!" << std::endl;
}
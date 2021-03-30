#pragma once

#include "sssp_fluid.h"
#include "sssp.h"

#include "matrix.h"
#include "matrix_fluid.h"

#include "kmeans_fluid.h"

#include "bellman_ford_fluid.h"

#include "graphcolor_fluid.h"

#include "sobel_fluid.h"

#include "cnn_fluid.h"

#include "medusa_fluid.h"

#include "blackscholes.h"

#include "fourier.h"

#include "dct.h"

#include "laplacian_fluid.h"

#include "test_out.h"

#include "../utils/timer.h"
#include <stdio.h>
#include <unistd.h>
#include <string>

extern Profiler GlobalProfiler;
extern CPUProfiler GlobalCPUProfiler;
#define THREADS_NUM 8


class Experiements {
public:
	std::string cmd;
	std::string inputfilename;
	std::string outputfilename;
	std::string experimentname;
	int threads_num;
	virtual void experiment() = 0;
	void print_log(std::string filename, std::vector<Guard*> guard_log) {
		FILE *f = fopen(filename.c_str(),"w");
		for (int i=0; i<guard_log.size(); i++) {
			Guard *g = guard_log[i];

			for(int j=0; j<g->logs.size(); j++) {
				fprintf(f, "%s\n", g->logs[j].c_str());
			}
			fprintf(f, "=========================================================================\n");
		}
		fclose(f);
	}
};

class KMeansExperiments : public Experiements {
	
public:
	static std::string inputpath;
	static std::string outputpath;
	KMeansExperiments(std::string filename) {
		inputfilename = filename;
	}
	double rate;
	int iterations;
	int class_num;
	int dups;
	virtual void experiment() {//singleton???
		int n = 1;
		std::vector<example::Clusters *> cvv;
		double cc = 0;
		double ss = 0;
		for (int i = 0; i < n; i++) {

			WallTimer wtdb; wtdb.start();
			CPUTimer  ctdb; ctdb.start();

			wtdb.start();
			ctdb.start();

			if(this->cmd == "KM") {
				auto km = new example::KmeansRandom();
				km -> rate = rate;
				km -> iterations = iterations;
				km -> class_num = class_num;
				km->kmeans(inputpath + inputfilename, outputpath);
	
				double tt=wtdb.stop();
				cc += tt;
				ss += km->sum;
				SyncLogger::print("Fluid CPUTimer: ", ctdb.stop(), " WallTimer: ", tt);
				//print_log("Kmeans_log", km->guard_log);
				FILE *f = fopen("output","a");
				fprintf(f,"%f %f\n",km->sum,tt);
				fclose(f);
				
				cvv.push_back(&(km->clusters));
			} else if(this->cmd == "KM_Fluid") {
				auto km = new example::KmeansSchemeFluidX_E();
				km -> rate = rate;
				km -> iterations = iterations;
				km -> class_num = class_num;
				km->kmeans(inputpath + inputfilename, outputpath);
	
				double tt=wtdb.stop();
				cc += tt;
				ss += km->sum;
				SyncLogger::print("Fluid CPUTimer: ", ctdb.stop(), " WallTimer: ", tt);
				//print_log("Kmeans_log", km->guard_log);
				FILE *f = fopen("output","a");
				fprintf(f,"%f %f\n",km->sum,tt);
				fclose(f);
				
				cvv.push_back(&(km->clusters));
			} else if(this->cmd == "KM_XS_Fluid") {
				auto km = new example::KmeansSchemeFluidXS_E();
				km -> rate = rate;
				km -> iterations = iterations;
				km -> class_num = class_num;
				km->kmeans(inputpath + inputfilename, outputpath);
	
				double tt=wtdb.stop();
				cc += tt;
				ss += km->sum;
				SyncLogger::print("Fluid CPUTimer: ", ctdb.stop(), " WallTimer: ", tt);
				//print_log("Kmeans_log", km->guard_log);
				FILE *f = fopen("output","a");
				fprintf(f,"%f %f\n",km->sum,tt);
				fclose(f);
				
				cvv.push_back(&(km->clusters));
			} else if(this->cmd == "KM_multi") {
				auto km = new example::KmeansRandomP(threads_num);
				km -> rate = rate;
				km -> iterations = iterations;
				km -> class_num = class_num;
				km->kmeans(inputpath + inputfilename, outputpath);
	
				double tt=wtdb.stop();
				cc += tt;
				ss += km->sum;
				SyncLogger::print("Fluid CPUTimer: ", ctdb.stop(), " WallTimer: ", tt);
				//print_log("Kmeans_log", km->guard_log);
				FILE *f = fopen("output","a");
				fprintf(f,"%f %f\n",km->sum,tt);
				fclose(f);
				
				cvv.push_back(&(km->clusters));
			} else if(this->cmd == "KM_multi_Fluid") {
				auto km = new example::KmeansSchemeFluidX2_E(threads_num);
				km -> rate = rate;
				km -> iterations = iterations;
				km -> class_num = class_num;
				km->kmeans(inputpath + inputfilename, outputpath);
	
				double tt=wtdb.stop();
				cc += tt;
				ss += km->sum;
				SyncLogger::print("Fluid CPUTimer: ", ctdb.stop(), " WallTimer: ", tt);
				//print_log("Kmeans_log", km->guard_log);
				FILE *f = fopen("output","a");
				fprintf(f,"%f %f\n",km->sum,tt);
				fclose(f);
				
				cvv.push_back(&(km->clusters));
			//auto km = new example::KmeansSchemeFluidX22_E(threads_num);
			}


			GlobalProfiler.printTimeStamp();
			GlobalCPUProfiler.printTimeStamp();
		}
		cc = cc / double(n);
		ss = ss / double(n);
		//FILE *f = fopen("WallTimer","a");
		//fprintf(f,"%f\n",cc);
		//fclose(f);
		FILE *f = fopen("DIST","a");
		fprintf(f," %10f %10f\n",ss,cc);
		fclose(f);
		std::cout << "Avg WallTimer: " << cc << " Avg DIST: " << ss << std::endl;
		example::Clusters ctotal;
		ctotal.k = cvv[0]->k; 
		ctotal.centroids = (example::Centroid*)malloc(ctotal.k * sizeof(example::Centroid));



		for (int i = 0; i < ctotal.k; i++) {
			(ctotal.centroids)[i].r = 0;
			(ctotal.centroids)[i].g = 0;
			(ctotal.centroids)[i].b = 0;
			(ctotal.centroids)[i].n = 0;
			for (auto &p : cvv) {
				(ctotal.centroids)[i].r += (p->centroids)[i].r;
				(ctotal.centroids)[i].g += (p->centroids)[i].g;
				(ctotal.centroids)[i].b += (p->centroids)[i].b;
				(ctotal.centroids)[i].n += (p->centroids)[i].n;
			}
		}
		for (int i = 0; i < ctotal.k; i++) {
			SyncLogger::print(
				"R:", (ctotal.centroids)[i].r,
				"G:", (ctotal.centroids)[i].g,
				"B:", (ctotal.centroids)[i].b,
				"N:", (ctotal.centroids)[i].n
			);
		}
		return;
	}
};



template<int const HH, int const WW>
class MatrixExperiment : public Experiements {
	static int hh, ww;
public:
	 virtual void experiment() {//singleton???

		auto me0 = new MatrixMultiplicationBaseline(hh, ww);
		auto me1 = new MatrixMultiplicationFluid(hh, ww);
		auto A = new Matrix<int>(hh, ww);
		auto B = new Matrix<int>(hh, ww);
		auto C = new Matrix<int>(hh, ww);
		auto D = new Matrix<int>(hh, ww, 0);

		WallTimer wtdb; wtdb.start();
		CPUTimer  ctdb; ctdb.start();
		//auto DB = me0->multi(A, B, C, D);
		//std::cout << "Baseline CPUTimer: " << ctdb.stop() << " WallTimer: " << wtdb.stop() <<std::endl;

		wtdb.start();
		ctdb.start();
		auto DF = me1->multi(A, B, C, D);
		SyncLogger::print("Fluid CPUTimer: ", ctdb.stop(), " WallTimer: ", wtdb.stop());
		//auto DF = me1->multi(A, B, C, D);

		//Matrix<int>::verify(DB, DF);
		GlobalProfiler.printTimeStamp();
		GlobalCPUProfiler.printTimeStamp();

		return;
	}
};

template<int const HH, int const WW>
int  MatrixExperiment<HH, WW>::hh = HH;

template<int const HH, int const WW>
int MatrixExperiment<HH, WW>::ww = WW;

class SSSPExperiment: public Experiements {
public:
	static std::string inputpath;
	typedef dragon_li::util::Types<
		int 			//SizeType
	> _Types;

	typedef dragon_li::util::Settings<
		_Types,						//types
		256, 						//THREADS
		104,						//CTAS
		5,							//CDP_THREADS_BITS
		32							//CDP_THRESHOLD
	> _Settings;

	typedef dragon_li::sssp::Types<
		_Types, 		//Basic Types
		int,			//VertexIdType
		int				//EdgeWeightType
	> Types;
	typedef dragon_li::sssp::Settings<
		_Settings, 					//Basic Settings
		Types,						//SSSP Types
		INT_MAX						//Max weight
	> Settings;
	virtual void experiment() override;
};

class Bellman_fordExperiments : public Experiements {
	
public:
	static std::string inputpath;
	static std::string outputpath;
	Bellman_fordExperiments(std::string filename) {
		inputfilename = filename;
	}
	double rate;
	virtual void experiment() {//singleton???
		int n = 1;
		double cc = 0;
		double ss = 0;
		for (int i = 0; i < n; i++) {

			WallTimer wtdb; wtdb.start();
			CPUTimer  ctdb; ctdb.start();

			wtdb.start();
			ctdb.start();
			if(this->cmd == "BF") {
				auto bf = new example::Bellman_ford();
				bf->rate = rate;
				bf -> bellman_ford(inputpath + inputfilename);
	
				double tt=wtdb.stop();
				//SyncLogger::print("Fluid CPUTimer: ", ctdb.stop(), " WallTimer: ", tt);
				printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
				//print_log("Bellmanford_log", bf->guard_log);
  				cc += tt;
				{
					FILE *fp = fopen("dist_result","w");
					for(int i=0; i<bf->graph->v_num; i++)
						fprintf(fp, "%d\n", bf->graph->dist[i].dist);
					fclose(fp);
				}
			} else if(this->cmd == "BF_Fluid") {
				auto bf = new example::Bellman_fordFluidX_E();
				bf->rate = rate;
				bf -> bellman_ford(inputpath + inputfilename);
	
				double tt=wtdb.stop();
				//SyncLogger::print("Fluid CPUTimer: ", ctdb.stop(), " WallTimer: ", tt);
				printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
				//print_log("Bellmanford_log", bf->guard_log);
  				cc += tt;
				{
					FILE *fp = fopen("dist_result","w");
					for(int i=0; i<bf->graph->v_num; i++)
						fprintf(fp, "%d\n", bf->graph->dist[i].dist);
					fclose(fp);
				}
			}
		}
		cc = cc/n;
		FILE *f = fopen("DIST","a");
		fprintf(f," %10f\n",cc);
		fclose(f);
		return;
	}
};

class GraphcolorExperiments : public Experiements {
	
public:
	static std::string inputpath;
	static std::string outputpath;
	GraphcolorExperiments(std::string filename) {
		inputfilename = filename;
	}
	double rate;
	int end_quality;
	virtual void experiment() {//singleton???
		int n = 1;
		double cc = 0;
		double ss = 0;
		for (int i = 0; i < n; i++) {

			WallTimer wtdb; wtdb.start();
			CPUTimer  ctdb; ctdb.start();

			wtdb.start();
			ctdb.start();
			if(this->cmd == "GC") {
				auto gc = new example::Graphcolor();
				gc->rate = rate;
				gc->end_quality = end_quality;
				gc -> graphcolor(inputpath + inputfilename);
			} else if(this->cmd == "GC_multi") {
				auto gc = new example::Graphcolor_multi(threads_num);
				gc->rate = rate;
				gc->end_quality = end_quality;
				gc -> graphcolor(inputpath + inputfilename);
			} else if(this->cmd == "GC_Fluid") {
				auto gc = new example::GraphcolorFluidX_E();
				gc->rate = rate;
				gc->end_quality = end_quality;
				gc -> graphcolor(inputpath + inputfilename);
			} else if(this->cmd == "GC_multi_Fluid") {
				auto gc = new example::GraphcolorFluidX_E_multi(threads_num);
				gc->rate = rate;
				gc->end_quality = end_quality;
				gc -> graphcolor(inputpath + inputfilename);
			}

			double tt=wtdb.stop();
			//SyncLogger::print("Fluid CPUTimer: ", ctdb.stop(), " WallTimer: ", tt);
			printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
			//print_log("Graphcolor_log", gc->guard_log);
		}
		return;
	}
};

class SobelExperiments : public Experiements {
	
public:
	static std::string inputpath;
	static std::string outputpath;
	SobelExperiments(std::string in, std::string out) {
		inputfilename = in;
		outputfilename = out;
	}
	double rate;
	double sobel_th;
	double sigma;
	int dups;
	virtual void experiment() {//singleton???
		int n = 1;
		double cc = 0; //time
		double ss = 0; //iou
		for (int i = 0; i < n; i++) {
 
			WallTimer wtdb; wtdb.start();
			CPUTimer  ctdb; ctdb.start();

			wtdb.start();
			ctdb.start();
			//auto sb = new example::Sobel();
			//auto sb = new example::Sobel_multi(threads_num);
			auto sb = new example::SobelFluid_E();
			//auto sb = new example::SobelFluid_multi_E(threads_num);
			sb->rate = rate;
			sb->sobel_th = sobel_th;
			sb->sigma = sigma;
			sb->dups = dups;
			sb -> sobel(inputpath + inputfilename, inputpath + outputfilename);

			double tt=wtdb.stop();
			printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
			////print_log("Sobel_log", sb->guard_log);
  			sb->testImage(sb->image, inputpath + outputfilename);
  			ss += sb->iou;
  			cc += tt;
		}
		ss = ss/n;
		cc = cc/n;
		FILE *f = fopen("DIST","a");
		fprintf(f," %10f %10f\n",ss,cc);
		fclose(f);
		return;
	}
};

class CNNExperiments : public Experiements {
	
public:
	static std::string inputpath;
	static std::string outputpath;
	CNNExperiments(std::string in) {
		inputfilename = in;
		//outputfilename = out;
	}
	double rate;
	double sigma;
	int dups;
	int batch_size;
	virtual void experiment() {//singleton???
		int n = 1;
		double cc = 0; //time
		double ss = 0; //iou
		for (int i = 0; i < n; i++) {

			WallTimer wtdb; wtdb.start();
			CPUTimer  ctdb; ctdb.start();

			wtdb.start();
			ctdb.start();
			std::cout << "before create a class\n";
			if(this->cmd == "CNN") {
				auto cn = new example::CNN_lenet(28, 28, 1, batch_size);
				cn->rate = rate;
				std::cout << "succ created a class\n";
				cn->cnn(inputpath, inputfilename);
				double tt=wtdb.stop();
				printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
	  			cn->test_result();
	  			ss += cn->accuracy;
	  			cc += tt;
			} else if(this->cmd == "CNN_Fluid") {
				auto cn = new example::CNN_lenetFluid_E(28, 28, 1, batch_size, rate);
				cn->rate = rate;
				std::cout << "succ created a class\n";
				cn->cnn(inputpath, inputfilename);
				double tt=wtdb.stop();
				printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
	  			cn->test_result();
	  			ss += cn->accuracy;
	  			cc += tt;
			} else if(this->cmd == "CNN_SqueezeNet") {
				auto cn = new example::CNN_squeezenet(28, 28, 1, batch_size);
				cn->rate = rate;
				std::cout << "succ created a class\n";
				cn->cnn(inputpath, inputfilename);
				double tt=wtdb.stop();
				printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
	  			cn->test_result();
	  			ss += cn->accuracy;
	  			cc += tt;
			}  else if(this->cmd == "CNN_SqueezeNet_Fluid") {
				auto cn = new example::CNN_squeezenetFluid_E(28, 28, 1, batch_size, rate);
				cn->rate = rate;
				std::cout << "succ created a class\n";
				cn->cnn(inputpath, inputfilename);
				double tt=wtdb.stop();
				printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
	  			cn->test_result();
	  			ss += cn->accuracy;
	  			cc += tt;
  			} else if(this->cmd == "CNN_VGGNet") {
				auto cn = new example::CNN_vggnet(224, 224, 3, batch_size);
				cn->rate = rate;
				std::cout << "succ created a class\n";
				cn->cnn(inputpath, inputfilename);
				double tt=wtdb.stop();
				printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
	  			cn->test_result();
	  			ss += cn->accuracy;
	  			cc += tt;
			} else if(this->cmd == "CNN_VGGNet_Fluid") {
				auto cn = new example::CNN_vggnetFluid_E(224, 224, 3, batch_size, rate);
				cn->rate = rate;
				std::cout << "succ created a class\n";
				cn->cnn(inputpath, inputfilename);
				double tt=wtdb.stop();
				printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
	  			cn->test_result();
	  			ss += cn->accuracy;
	  			cc += tt;
			}
		}
		ss = ss/n;
		cc = cc/n;
		FILE *f = fopen("DIST","a");
		fprintf(f," %10f %10f\n",ss,cc);
		fclose(f);
		return;
	}
};

class MedusaExperiments : public Experiements {
	
public:
	static std::string inputpath;
	static std::string outputpath;
	MedusaExperiments(std::string in) {
		inputfilename = in;
		//outputfilename = out;
	}
	double rate;
	double sigma;
	int dups;
	int iterations;
	int select_num;
	int rmsd_th;
	virtual void experiment() {//singleton???
		int n = 1;
		double cc = 0; //time
		double ss = 0; //iou
		for (int i = 0; i < n; i++) {

			WallTimer wtdb; wtdb.start();
			CPUTimer  ctdb; ctdb.start();

			wtdb.start();
			ctdb.start();
			std::cout << "before create a class\n";
			if(this->cmd == "CNN") {
				auto md = new example::Medusa(iterations, select_num, rmsd_th);
				md->rate = double(rate) / 100;
				std::cout << "succ created a class\n";
				md->medusa(inputpath, inputfilename);
  				ss += md->accuracy;
			} else if(this->cmd == "MD_Fluid") {
				auto md = new example::MedusaFluid_E(iterations, select_num, rmsd_th);
				md->rate = double(rate) / 100;
				std::cout << "succ created a class\n";
				md->medusa(inputpath, inputfilename);
  				ss += md->accuracy;
			}

			double tt=wtdb.stop();
			//SyncLogger::print("Fluid CPUTimer: ", ctdb.stop(), " WallTimer: ", tt);
			printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
  			//print_log("Medusa_log", md->guard_log);
			//sleep(1);

  			cc += tt;
		}
		ss = ss/n;
		cc = cc/n;
		FILE *f = fopen("DIST","a");
		fprintf(f," %10f %10f\n",ss,cc);
		fclose(f);
		return;
	}
};

class BlackscholesExperiments : public Experiements {
	
public:
	static std::string inputpath;
	static std::string outputpath;
	BlackscholesExperiments(std::string in) {
		inputfilename = in;
		//outputfilename = out;
	}
	double rate;
	double sigma;
	int dups;
	int select_num;
	virtual void experiment() {//singleton???
		int n = 1;
		double cc = 0; //time
		double ss = 0; //iou
		for (int i = 0; i < n; i++) {

			WallTimer wtdb; wtdb.start();
			CPUTimer  ctdb; ctdb.start();

			wtdb.start();
			ctdb.start();
			std::cout << "before create a class\n";
			auto md = new example::Blackscholes(select_num);
			//auto md = new example::BlackscholesFluid_E(iterations, select_num, rmsd_th);
			//auto md = new example::CNNFluid_E(1, 1, 784, iterations, rate);
			md->fluid_rate = rate;
			std::cout << "succ created a class\n";

			md->blackscholes(inputpath, inputfilename);

			double tt=wtdb.stop();
			//SyncLogger::print("Fluid CPUTimer: ", ctdb.stop(), " WallTimer: ", tt);
			printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
  			//print_log("Blackscholes_log", md->guard_log);
			//sleep(1);

  			ss += md->accuracy;
  			cc += tt;
		}
		ss = ss/n;
		cc = cc/n;
		FILE *f = fopen("DIST","a");
		fprintf(f," %10f %10f\n",ss,cc);
		fclose(f);
		return;
	}
};

class FFTExperiments : public Experiements {

public:
        double rate;
        int f1;
        FFTExperiments(){
}
        virtual void experiment() {//singleton???
                int n = 1;
                for (int i = 0; i < n; i++) {

                        WallTimer wtdb; wtdb.start();
                        CPUTimer  ctdb; ctdb.start();

                        wtdb.start();
                        ctdb.start();

                        auto *fft = new example::fftFluidX_E();
                        //auto gc = new example::GraphcolorFluidX();
                        //auto gc = new example::GraphcolorFluidXS();
                        //auto gc = new example::GraphcolorFluidX_E();
                        fft->rate = rate;
                        fft->n = f1;
                        fft->thread_num = threads_num;
                        //fft ->FftFluidX_E(f1);
                        fft->Fft(f1, this->cmd);
                        double tt=wtdb.stop();
                        //SyncLogger::print("Fluid CPUTimer: ", ctdb.stop(), " WallTimer: ", tt);
						printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
                        //print_log("fft_log", fft->guard_log);
                }
                return;
        }
};


class DCTExperiments : public Experiements {

public:
        double rate;
        int rows;
        int columns;
        int pixel;

        DCTExperiments(){
}
        virtual void experiment() {//singleton???
                int n = 1;
                for (int i = 0; i < n; i++) {

                        WallTimer wtdb; wtdb.start();
                        CPUTimer  ctdb; ctdb.start();

                        wtdb.start();
                        ctdb.start();

                        auto *dct = new example::dctFluidX_E();
                        //auto gc = new example::GraphcolorFluidX();
                        //auto gc = new example::GraphcolorFluidXS();
                        //auto gc = new example::GraphcolorFluidX_E();
                        dct->rate = rate;
                        dct->m=rows;
                        dct->n=columns;
                        dct->a=pixel;
                        dct->dct_calc();
                        //dct->no_fluid();
                        double tt=wtdb.stop();
                        //SyncLogger::print("Fluid CPUTimer: ", ctdb.stop(), " WallTimer: ", tt);
						printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
                }
                return;
        }
};


class LaplacianExperiments : public Experiements {
	
public:
	static std::string inputpath;
	static std::string outputpath;
	LaplacianExperiments(std::string in, std::string out) {
		inputfilename = in;
		outputfilename = out;
	}
	double rate;
	double sobel_th;
	double sigma;
	int dups;
	int mode;
	virtual void experiment() {//singleton???
		int n = 1;
		double cc = 0; //time
		double ss = 0; //iou
		for (int i = 0; i < n; i++) {
 
			WallTimer wtdb; wtdb.start();
			CPUTimer  ctdb; ctdb.start();

			wtdb.start();
			ctdb.start();

			if(this->cmd == "ED" || this->cmd == "ED_GS" || this->cmd == "ED_GL" || this->cmd == "ED_MS" || this->cmd == "ED_ML") {
				auto lp = new example::Laplacian();
				lp->rate = rate;
				lp->sobel_th = sobel_th;
				lp->sigma = sigma;
				lp->dups = dups;
				lp->mode = mode;
				lp -> sobel(inputpath + inputfilename, inputpath + outputfilename);
	
				double tt=wtdb.stop();
				printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
				////print_log("Sobel_log", lp->guard_log);
				//sleep(1);
  				lp->testImage(lp->image, inputpath + outputfilename);
  				ss += lp->iou;
  				cc += tt;
			} else if (this->cmd == "ED_Fluid" || this->cmd == "ED_GS_Fluid" || this->cmd == "ED_GL_Fluid" || this->cmd == "ED_MS_Fluid" || this->cmd == "ED_ML_Fluid") {
				auto lp = new example::LaplacianFluid_E();
				lp->rate = rate;
				lp->sobel_th = sobel_th;
				lp->sigma = sigma;
				lp->dups = dups;
				lp->mode = mode;
				lp -> sobel(inputpath + inputfilename, inputpath + outputfilename);
	
				double tt=wtdb.stop();
				printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
				////print_log("Sobel_log", lp->guard_log);
				//sleep(1);
  				lp->testImage(lp->image, inputpath + outputfilename);
  				ss += lp->iou;
  				cc += tt;
			} else if (this->cmd == "ED_multi") {
				auto lp = new example::Laplacian_multi(threads_num);
				lp->rate = rate;
				lp->sobel_th = sobel_th;
				lp->sigma = sigma;
				lp->dups = dups;
				lp->mode = mode;
				lp -> sobel(inputpath + inputfilename, inputpath + outputfilename);
	
				double tt=wtdb.stop();
				printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
				////print_log("Sobel_log", lp->guard_log);
				//sleep(1);
  				lp->testImage(lp->image, inputpath + outputfilename);
  				ss += lp->iou;
  				cc += tt;
			} else if (this->cmd == "ED_multi_Fluid") {
				auto lp = new example::LaplacianFluid_multi_E(threads_num);
				lp->rate = rate;
				lp->sobel_th = sobel_th;
				lp->sigma = sigma;
				lp->dups = dups;
				lp->mode = mode;
				lp -> sobel(inputpath + inputfilename, inputpath + outputfilename);
	
				double tt=wtdb.stop();
				printf("Fluid CPUTimer: %10f WallTimer: %10f\n", ctdb.stop(), tt);
				////print_log("Sobel_log", lp->guard_log);
				//sleep(1);
  				lp->testImage(lp->image, inputpath + outputfilename);
  				ss += lp->iou;
  				cc += tt;
			}
		}
		ss = ss/n;
		cc = cc/n;
		FILE *f = fopen("DIST","a");
		fprintf(f," %10f %10f\n",ss,cc);
		fclose(f);
		return;
	}
};


class TestExperiments : public Experiements {
	
public:
	static std::string inputpath;
	static std::string outputpath;
	TestExperiments(std::string filename) {
		inputfilename = filename;
	}
	double rate;
	int iterations;
	int class_num;
	virtual void experiment() {//singleton???
		int n = 1;
		std::vector<example::Clusters *> cvv;
		double cc = 0;
		double ss = 0;
		for (int i = 0; i < n; i++) {

			WallTimer wtdb; wtdb.start();
			CPUTimer  ctdb; ctdb.start();

			wtdb.start();
			ctdb.start();
			//auto km = new example::KmeansSchemeFluidX();
			auto km = new example::KmeansImage();
			

			km -> rate = rate;
			km -> iterations = iterations;
			km -> class_num = class_num;
			km->kmeans(inputpath + inputfilename, outputpath);

			double tt=wtdb.stop();
			cc += tt;
			ss += km->sum;
			SyncLogger::print("Fluid CPUTimer: ", ctdb.stop(), " WallTimer: ", tt);
		}
		cc = cc / double(n);
		ss = ss / double(n);
		std::cout << "Avg WallTimer: " << cc << " Avg DIST: " << ss << std::endl;

		return;
	}
};

#include "kmeans.h"
//Haibo, Huaipan

#include "kmeans_fluid.h"
using RgbImage = example::RgbImage;
using Clusters = example::Clusters;
using namespace example;

std::string Kmeans::loopindicator = "0";
void KmeansSchemeFluidExtreme::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(12);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	auto tp3 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp3status = new __fluid__<bool>(tp3->isfinished());
	auto v3 = new ValveEQ<bool>(tp3status, true);

	tp0->run();
	tp3->run();

	double  rate = 1;

	for (int i = 0; i < n; ++i) {
		int* counter = new int(0);
		auto tpb1 = std::bind(&KmeansRandom::AssignCluster, this, image, clusters, std::placeholders::_1);
		auto tp1 = tf->newTask<decltype(tpb1), int*>("AssignCluster" + std::to_string(i), tpb1, counter);
		Guard* g1 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { }, tp1, {});

		auto tp1status = new __fluid__<int>(counter);

		if (rate <= 0.05) rate = 0.05;

		auto v1 = new ValveGT<int>(tp1status, image->h*image->w*rate);

		auto tpb2 = std::bind(&KmeansRandom::Recenter, this, image, clusters);
		auto tp2 = tf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), tpb2);
		Guard*       g2 = gs->newGuard("GuardRecenter" + std::to_string(i), {  }, tp2, { g1 });

		tp0status = new __fluid__<bool>(tp2->isfinished());
		v0 = new ValveEQ<bool>(tp0status, true);
		tp3status = new __fluid__<bool>(tp1->isfinished());
		v3 = new ValveEQ<bool>(tp3status, true);
		rate = rate - i *0.025;
	}

	tp3 = tf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	gs->sync(tp3);
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}



void KmeansSchemeFluidXI::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(2);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	auto tp3 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp3status = new __fluid__<bool>(tp3->isfinished());
	auto v3 = new ValveEQ<bool>(tp3status, true);

	tp0->run();
	tp3->run();

	double  rate = 1;

	for (int i = 0; i < n; ++i) {
		int* counter = new int(0);
		auto tpb1 = std::bind(&KmeansRandom::AssignCluster, this, image, clusters, std::placeholders::_1);
		auto tp1 = tf->newTask<decltype(tpb1), int*>("AssignCluster" + std::to_string(i), tpb1, counter);
		Guard* g1 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0, v3 }, tp1, {});

		auto tp1status = new __fluid__<int>(counter);

		if (rate <= 0.05) rate = 0.05;

		auto v1 = new ValveGT<int>(tp1status, image->h*image->w*rate);

		auto tpb2 = std::bind(&KmeansRandom::Recenter, this, image, clusters);
		auto tp2 = tf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), tpb2);
		Guard*       g2 = gs->newGuard("GuardRecenter" + std::to_string(i), { v1 }, tp2, { g1 });

		tp0status = new __fluid__<bool>(tp2->isfinished());
		v0 = new ValveEQ<bool>(tp0status, true);
		tp3status = new __fluid__<bool>(tp1->isfinished());
		v3 = new ValveEQ<bool>(tp3status, true);
		rate = rate - i *0.025;
	}

	tp3 = tf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	v3 = new ValveEQ<bool>(tp3status, true);
	while (*(tp3->parent->isfinished()) == false)
		; // HARD HACK!

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}
void KmeansSchemeFluidPause::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	auto tf = new TaskFactory();
	auto gs = new AggressiveGSPause(8);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

	double  rate = 1;

	for (int i = 0; i < n; ++i) {
		loopindicator = "loop" + std::to_string(i);
		int* counter = new int(0);
		auto tpb1 = std::bind(&KmeansRandom::AssignCluster, this, image, clusters, std::placeholders::_1);
		auto tp1 = tf->newTask<decltype(tpb1), int*>("AssignCluster" + std::to_string(i), tpb1, counter);
		Guard* g1 = gs->newGuard("GuardAssignCluster" + std::to_string(i), {/*v0*/ }, tp1, {});

		auto tp1status = new __fluid__<int>(counter);
		//if (i < 10) rate = 1;
		if (rate <= 0.05) rate = 0.05;

		auto v1 = new ValveGT<int>(tp1status, image->h*image->w*rate);

		auto tpb2 = std::bind(&KmeansRandom::Recenter, this, image, clusters);
		auto tp2 = tf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), tpb2);
		Guard*       g2 = gs->newGuard("GuardRecenter" + std::to_string(i), { v1 }, tp2, { g1 });

		gs->iSync();
		//tp0status = new __fluid__<bool>(tp2->isfinished());
		//v0 = new ValveEQ<bool>(tp0status, true);
		rate = rate - i *1.0/n;
	}

	auto tp3 = tf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	auto v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	gs->sync(tp3);
	double sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	SyncLogger::print("DIST: ", sum);
	
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}
void KmeansSchemeFluidX::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	ttf = new TaskFactory();
	ggs = new AggressiveGS(4);
	auto tp0 = ttf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

	double  rate = 1;
		//rate = 0.1;
	//ValveGT<int> *v2;
	rate = this->rate;
	SyncLogger::print("=====KmeansSchemeFluidX=====");
	//time_tag.resize(n*10);
	double pre = 100000000.0;
	double sum = 0.0;
	for (int i = 0; i < n; ++i) {
		loopindicator = "loop" + std::to_string(i);
		int* counter = new int(0);
		auto tpb1 = std::bind(&KmeansSchemeFluidX::AssignCluster, this, image, clusters, std::placeholders::_1);
		auto tp1 = ttf->newTask<decltype(tpb1), int*>("AssignCluster" + std::to_string(i), tpb1, counter);
		Guard* g1 = ggs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1, {});
		//std::cout << "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFrom Recenter " << i << ": " << counter << std::endl;
		//std::cout << "" << 

		auto tp1status = new __fluid__<int>(counter);

		//if (i < 10) rate = 1;
		//if (rate <= 0.025) rate = 0.025;
		//rate = 0.1;

		auto v1 = new ValveGT<int>(tp1status, image->h*image->w*rate);

		auto tpb2 = std::bind(&KmeansSchemeFluidX::Recenter, this, image, clusters);
		auto tp2 = ttf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), tpb2);
		Guard*       g2 = ggs->newGuard("GuardRecenter" + std::to_string(i), { v1 }, tp2, {  });
		//gs->iSync();
		//gs->sync(tp2);
		tp0status = new __fluid__<bool>(tp2->isfinished());
		v0 = new ValveEQ<bool>(tp0status, true);
		//std::cout << "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFrom Recenter " << i << ": " << v0 << std::endl;
		//v2 = v1;
		//rate = rate - i  *1.0 / n;
		//rate = 0.1;

		//SyncLogger::print("DIST: ", sum);
		//this->sum = sum;

		//conversion
		//sum = print_sum_to_DIST1(image, clusters);
		//if(sum > pre - 0.01)
		//{
		//	std::cout << "terminate at iter: " << i << std::endl;
		//	break;
		//}
		//pre = sum;
	}

	auto tp3 = ttf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	auto v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	ggs->sync(tp3);

	sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	sum = print_sum_to_DIST1(image, clusters);
	SyncLogger::print("DIST: ", sum);
	this->sum = sum;

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}
void KmeansSchemeFluidXS::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(4);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

	double  rate = this->rate;
	//double rate = 0.8;
	std::cout << rate << std::endl;
	SyncLogger::print("=====KmeansSchemeFluidX=====");

	for (int i = 0; i < n; ++i) {
		loopindicator = "loop" + std::to_string(i);
		int* counter = new int(0);
		auto tpb1 = std::bind(&KmeansSchemeFluidXS::AssignCluster_stable, this, image, clusters, n, std::placeholders::_1);
		auto tp1 = tf->newTask<decltype(tpb1), int*>("AssignCluster" + std::to_string(i), tpb1, counter);
		Guard* g1 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1, {});

		auto tp1status = new __fluid__<int>(counter);

		//if (i < 10) rate = 1;
		//if (rate <= 0.025) rate = 0.025;

		auto v1 = new ValveGT<int>(tp1status, image->h*image->w*rate);

		auto tpb2 = std::bind(&KmeansSchemeFluidXS::Recenter, this, image, clusters);
		auto tp2 = tf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), tpb2);
		Guard*       g2 = gs->newGuard("GuardRecenter" + std::to_string(i), { v1 }, tp2, {});
		//gs->iSync();
		//gs->sync(tp2);
		
		tp0status = new __fluid__<bool>(tp2->isfinished());
		v0 = new ValveEQ<bool>(tp0status, true);
		//rate = rate - i  *1.0 / n;
	}

	auto tp3 = tf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	auto v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	gs->sync(tp3);

	double sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	SyncLogger::print("DIST: ", sum);
	this->sum = sum;

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}
void KmeansSchemeFluidXSB::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(4);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

	double  rate = this->rate;
	//double rate = 0.8;
	std::cout << rate << std::endl;
	SyncLogger::print("=====KmeansSchemeFluidX=====");

	for (int i = 0; i < n; ++i) {
		loopindicator = "loop" + std::to_string(i);
		int* counter = new int(0);
		auto tp1status = new __fluid__<int>(counter);
		auto tpb1 = std::bind(&KmeansSchemeFluidXS::AssignCluster_sb, this, image, clusters, n, std::placeholders::_1);
		auto tp1 = tf->newTask<decltype(tpb1), int*>("AssignCluster" + std::to_string(i), tpb1, counter);
		Guard* g1 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1, {});


		//if (i < 10) rate = 1;
		//if (rate <= 0.025) rate = 0.025;

		//auto v1_ = new ValveST<int>(tp1status, image->h*image->w*rate);
		//auto v1_ = v1.init(tp1status, image->h*image->w*rate);
		//Kmeans **xxx = new Kmeans*(this);
		//*xxx = this;
		auto sbstatus = new __fluid__<Kmeans*>(new Kmeans*(this));
		auto v1_ = v21.init(sbstatus,tp1->isfinished(),i+1);

		auto tpb2 = std::bind(&KmeansSchemeFluidXS::Recenter, this, image, clusters);
		auto tp2 = tf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), tpb2);
		Guard*       g2 = gs->newGuard("GuardRecenter" + std::to_string(i), { v1_ }, tp2, {});
		//gs->iSync();
		//gs->sync(tp2);
		
		tp0status = new __fluid__<bool>(tp2->isfinished());
		v0 = new ValveEQ<bool>(tp0status, true);
		//rate = rate - i  *1.0 / n;
	}

	auto tp3 = tf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	auto v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	gs->sync(tp3);

	double sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	SyncLogger::print("DIST: ", sum);
	this->sum = sum;

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}
void KmeansSchemeFluidXpq::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(2);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);


	tp0->run();

	double  rate = 1;
	rate = this->rate;
	SyncLogger::print("=====KmeansSchemeFluidX=====");

	for (int i = 0; i < n; ++i) {
		loopindicator = "loop" + std::to_string(i);
		int* counter = new int(0);
		auto tpb1 = std::bind(&KmeansRandom::AssignCluster_pq, this, image, clusters, n, std::placeholders::_1);
		auto tp1 = tf->newTask<decltype(tpb1), int*>("AssignCluster" + std::to_string(i), tpb1, counter);
		Guard* g1 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { /*v0*/ }, tp1, {});

		auto tp1status = new __fluid__<int>(counter);

		//if (i < 10) rate = 1;
		//if (rate <= 0.025) rate = 0.025;
		//rate = 0.1;

		auto v1 = new ValveGT<int>(tp1status, image->h*image->w*rate);

		auto tpb2 = std::bind(&KmeansRandom::Recenter, this, image, clusters);
		auto tp2 = tf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), tpb2);
		Guard*       g2 = gs->newGuard("GuardRecenter" + std::to_string(i), { v1 }, tp2, { g1 });
		gs->iSync();
		//tp0status = new __fluid__<bool>(tp2->isfinished());
		//v0 = new ValveEQ<bool>(tp0status, true);
		//rate = rate - i  *1.0 / n;
		//rate = 0.1;
	}

	auto tp3 = tf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	auto v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	gs->sync(tp3);

	double sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	SyncLogger::print("DIST: ", sum);
	this->sum = sum;

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}

void KmeansSchemeFluidX2::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(12);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

	double  rate = 1;
	rate = this->rate;
	SyncLogger::print("=====KmeansSchemeFluidX2 thread_num:%d=====", this->thread_num);

	for (int i = 0; i < n; ++i) {
		loopindicator = "loop" + std::to_string(i);

/*
 		int *call_num1__anything = new int(0);
		auto call_num1 = new __fluid__<int>(call_num1__anything);

		auto tpb1_0 = std::bind(&KmeansRandom::AssignCluster2, this, image, clusters, this->thread_num, 0, std::placeholders::_1);
		auto tp1_0 = tf->newTask<decltype(tpb1_0), int*>("AssignCluster" + std::to_string(i), tpb1_0, call_num1->p);
		Guard* g1_0 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_0, {});

		auto tpb1_1 = std::bind(&KmeansRandom::AssignCluster2, this, image, clusters, this->thread_num, 1, std::placeholders::_1);
		auto tp1_1 = tf->newTask<decltype(tpb1_1), int*>("AssignCluster" + std::to_string(i), tpb1_1, call_num1->p);
		Guard* g1_1 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_1, {});
		
		auto tpb1_2 = std::bind(&KmeansRandom::AssignCluster2, this, image, clusters, this->thread_num, 2, std::placeholders::_1);
		auto tp1_2 = tf->newTask<decltype(tpb1_2), int*>("AssignCluster" + std::to_string(i), tpb1_2, call_num1->p);
		Guard* g1_2 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_2, {});
		
		auto tpb1_3 = std::bind(&KmeansRandom::AssignCluster2, this, image, clusters, this->thread_num, 3, std::placeholders::_1);
		auto tp1_3 = tf->newTask<decltype(tpb1_3), int*>("AssignCluster" + std::to_string(i), tpb1_3, call_num1->p);
		Guard* g1_3 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_3, {});

		auto v0_ = v1.init(call_num1,image->size*rate); */
		//auto tp1status = new __fluid__<std::vector<int>>(counter);

		//if (i < 10) rate = 1;
		//if (rate <= 0.025) rate = 0.025;

		//auto v1 = new ValveGT_vec<int>(tp1status, image->h*image->w*rate);

		////#pragma call_num{__call_num__ call_num1 this->thread_num}
 		std::vector<int> *call_num1__anything = new std::vector<int>;
 		call_num1__anything->resize(this->thread_num);
		auto call_num1 = new __fluid__<std::vector<int>>(call_num1__anything); //&((*(call_num1->p))[i])

		////<<<g1,{v0},{},call_num1,this->thread_num>>>AssignCluster2(image, clusters, this->thread_num);
		std::set<Guard*> g1;
		for(int th=0; th<this->thread_num; th++)
		{
			auto tpb1_0 = std::bind(&KmeansSchemeFluidX2::AssignCluster2, this, image, clusters, this->thread_num, th, std::placeholders::_1);
			auto tp1_0 = tf->newTask<decltype(tpb1_0), std::vector<int>*>("AssignCluster" + std::to_string(i), tpb1_0, call_num1->p);
			Guard* g1_0 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_0, {});
			g1.insert(g1_0);
		}
		auto v0_ = v2.init(call_num1, image->h*image->w*rate);

		//auto tpb2 = std::bind(&KmeansRandom::Recenter2, this, image, clusters, this->thread_num);
		//auto tpb2 = std::bind(&KmeansRandomChangedRecenter::Recenter2, this, image, clusters, this->thread_num);
		auto tpb2 = std::bind(&KmeansSchemeFluidX2::Recenter, this, image, clusters);
		auto tp2 = tf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), tpb2);
		Guard*       g2 = gs->newGuard("GuardRecenter" + std::to_string(i), { v0_ }, tp2, {  });
		//gs->iSync();
		tp0status = new __fluid__<bool>(tp2->isfinished());
		v0 = new ValveEQ<bool>(tp0status, true);
		//rate = rate - i  *1.0 / n;
	}

	//auto tp3 = tf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	//auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	//auto v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	gs->sync(tp0);

	double sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	SyncLogger::print("DIST: ", sum);
	this->sum = sum;

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}
void KmeansSchemeFluidX2S::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(12);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

	double  rate = 1;
	rate = this->rate;
	SyncLogger::print("=====KmeansSchemeFluidX2 thread_num:%d=====", this->thread_num);

	for (int i = 0; i < n; ++i) {
		loopindicator = "loop" + std::to_string(i);
		//int* counter = new int(0);
		//std::vector<int>* counter = new std::vector<int>;
		//counter->push_back(0);
		//counter->insert(counter->begin(), this->thread_num-1, 0);

 		//int *call_num1__anything = new int(0);
		//auto call_num1 = new __fluid__<int>(call_num1__anything);
		////printf("counter size: %d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",counter->size());
		//auto tpb1 = std::bind(&KmeansRandom::AssignCluster2_stable, this, image, clusters, this->thread_num, std::placeholders::_1);
		//auto tp1 = tf->newTask<decltype(tpb1), int*>("AssignCluster" + std::to_string(i), tpb1, call_num1->p);
		//Guard* g1 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1, {});

		//auto tp1status = new __fluid__<std::vector<int>>(counter);

		//if (i < 10) rate = 1;
		//if (rate <= 0.025) rate = 0.025;

		//auto v1 = new ValveGT_vec<int>(tp1status, image->h*image->w*rate);
		////#pragma call_num{__call_num__ call_num1 this->thread_num}
 		std::vector<int> *call_num1__anything = new std::vector<int>;
 		call_num1__anything->resize(this->thread_num);
		auto call_num1 = new __fluid__<std::vector<int>>(call_num1__anything); //&((*(call_num1->p))[i])

		////<<<g1,{v0},{},call_num1,this->thread_num>>>AssignCluster2_stable(image, clusters, this->thread_num);
		std::set<Guard*> g1;
		for(int th=0; th<this->thread_num; th++)
		{
			auto tpb1_0 = std::bind(&KmeansSchemeFluidX2S::AssignCluster2_stable, this, image, clusters, this->thread_num, th, std::placeholders::_1);
			auto tp1_0 = tf->newTask<decltype(tpb1_0), std::vector<int>*>("AssignCluster" + std::to_string(i), tpb1_0, call_num1->p);
			Guard* g1_0 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_0, {});
			g1.insert(g1_0);
		}

		//auto tpb1_0 = std::bind(&KmeansSchemeFluidX2S::AssignCluster2_stable, this, image, clusters, this->thread_num, th, std::placeholders::_1);
		//auto tp1_0 = tf->newTask<decltype(tpb1_0), std::vector<int>*>("AssignCluster" + std::to_string(i), tpb1_0, call_num1->p);
		//Guard* g1_0 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_0, {});
		//auto tpb1_1 = std::bind(&KmeansSchemeFluidX2S::AssignCluster2_stable, this, image, clusters, this->thread_num, th, std::placeholders::_1);
		//auto tp1_1 = tf->newTask<decltype(tpb1_1), std::vector<int>*>("AssignCluster" + std::to_string(i), tpb1_1, call_num1->p);
		//Guard* g1_1 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_0, {});
		//auto tpb1_2 = std::bind(&KmeansSchemeFluidX2S::AssignCluster2_stable, this, image, clusters, this->thread_num, th, std::placeholders::_1);
		//auto tp1_2 = tf->newTask<decltype(tpb1_2), std::vector<int>*>("AssignCluster" + std::to_string(i), tpb1_2, call_num1->p);
		//Guard* g1_2 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_0, {});
		//auto tpb1_3 = std::bind(&KmeansSchemeFluidX2S::AssignCluster2_stable, this, image, clusters, this->thread_num, th, std::placeholders::_1);
		//auto tp1_3 = tf->newTask<decltype(tpb1_3), std::vector<int>*>("AssignCluster" + std::to_string(i), tpb1_3, call_num1->p);
		//Guard* g1_3 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_0, {});

		//auto v0_ = v1.init(call_num1,image->size*rate/this->thread_num);

		//auto tpb2 = std::bind(&KmeansRandom::Recenter2, this, image, clusters, this->thread_num);
		//auto tpb2 = std::bind(&KmeansRandomChangedRecenter::Recenter2, this, image, clusters, this->thread_num);

		////<<<g2,{v0},g1,call_num1,this->thread_num>>>Recenter(image, clusters);
		auto v0_ = v2.init(call_num1, image->h*image->w*rate);
		auto tpb2 = std::bind(&KmeansSchemeFluidX2S::Recenter, this, image, clusters);
		auto tp2 = tf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), tpb2);
		Guard*       g2 = gs->newGuard("GuardRecenter" + std::to_string(i), { v0_ }, tp2, {});
		//gs->iSync();
		tp0status = new __fluid__<bool>(tp2->isfinished());
		v0 = new ValveEQ<bool>(tp0status, true);
		//rate = rate - i  *1.0 / n;
	}

	auto tp3 = tf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	auto v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	gs->sync(tp3);

	double sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	SyncLogger::print("DIST: ", sum);
	this->sum = sum;

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}


void KmeansSchemeFluidX2SB::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(12);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

	double  rate = 1;
	rate = this->rate;
	SyncLogger::print("=====KmeansSchemeFluidX2 thread_num:%d=====", this->thread_num);

	for (int i = 0; i < n; ++i) {
		loopindicator = "loop" + std::to_string(i);
		//int* counter = new int(0);
		//std::vector<int>* counter = new std::vector<int>;
		//counter->push_back(0);
		//counter->insert(counter->begin(), this->thread_num-1, 0);

 		//int *call_num1__anything = new int(0);
		//auto call_num1 = new __fluid__<int>(call_num1__anything);
		////printf("counter size: %d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",counter->size());
		//auto tpb1 = std::bind(&KmeansRandom::AssignCluster2_stable, this, image, clusters, this->thread_num, std::placeholders::_1);
		//auto tp1 = tf->newTask<decltype(tpb1), int*>("AssignCluster" + std::to_string(i), tpb1, call_num1->p);
		//Guard* g1 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1, {});

		//auto tp1status = new __fluid__<std::vector<int>>(counter);

		//if (i < 10) rate = 1;
		//if (rate <= 0.025) rate = 0.025;

		//auto v1 = new ValveGT_vec<int>(tp1status, image->h*image->w*rate);
		////#pragma call_num{__call_num__ call_num1 this->thread_num}
 		std::vector<int> *call_num1__anything = new std::vector<int>;
 		call_num1__anything->resize(this->thread_num);
		auto call_num1 = new __fluid__<std::vector<int>>(call_num1__anything); //&((*(call_num1->p))[i])

		////<<<g1,{v0},{},call_num1,this->thread_num>>>AssignCluster2_stable(image, clusters, this->thread_num);
		std::set<Guard*> g1;
		for(int th=0; th<this->thread_num; th++)
		{
			auto tpb1_0 = std::bind(&KmeansSchemeFluidX2SB::AssignCluster2_sb, this, image, clusters, this->thread_num, th, std::placeholders::_1);
			auto tp1_0 = tf->newTask<decltype(tpb1_0), std::vector<int>*>("AssignCluster" + std::to_string(i), tpb1_0, call_num1->p);
			Guard* g1_0 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_0, {});
			g1.insert(g1_0);
		}

		//auto tpb1_0 = std::bind(&KmeansSchemeFluidX2S::AssignCluster2_stable, this, image, clusters, this->thread_num, th, std::placeholders::_1);
		//auto tp1_0 = tf->newTask<decltype(tpb1_0), std::vector<int>*>("AssignCluster" + std::to_string(i), tpb1_0, call_num1->p);
		//Guard* g1_0 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_0, {});
		//auto tpb1_1 = std::bind(&KmeansSchemeFluidX2S::AssignCluster2_stable, this, image, clusters, this->thread_num, th, std::placeholders::_1);
		//auto tp1_1 = tf->newTask<decltype(tpb1_1), std::vector<int>*>("AssignCluster" + std::to_string(i), tpb1_1, call_num1->p);
		//Guard* g1_1 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_0, {});
		//auto tpb1_2 = std::bind(&KmeansSchemeFluidX2S::AssignCluster2_stable, this, image, clusters, this->thread_num, th, std::placeholders::_1);
		//auto tp1_2 = tf->newTask<decltype(tpb1_2), std::vector<int>*>("AssignCluster" + std::to_string(i), tpb1_2, call_num1->p);
		//Guard* g1_2 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_0, {});
		//auto tpb1_3 = std::bind(&KmeansSchemeFluidX2S::AssignCluster2_stable, this, image, clusters, this->thread_num, th, std::placeholders::_1);
		//auto tp1_3 = tf->newTask<decltype(tpb1_3), std::vector<int>*>("AssignCluster" + std::to_string(i), tpb1_3, call_num1->p);
		//Guard* g1_3 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_0, {});

		//auto v0_ = v1.init(call_num1,image->size*rate/this->thread_num);

		//auto tpb2 = std::bind(&KmeansRandom::Recenter2, this, image, clusters, this->thread_num);
		//auto tpb2 = std::bind(&KmeansRandomChangedRecenter::Recenter2, this, image, clusters, this->thread_num);

		////<<<g2,{v0},g1,call_num1,this->thread_num>>>Recenter(image, clusters);
		//auto v0_ = v2.init(call_num1, image->h*image->w*rate);

		Kmeans *xxx = this;
		auto sbstatus = new __fluid__<Kmeans*>(&xxx);
		std::vector<bool*> list_finish;
		for (auto it = g1.begin(); it != g1.end(); it ++)
			list_finish.push_back((*it)->task->isfinished());
		auto v1_ = v20.init(sbstatus,list_finish);

		auto tpb2 = std::bind(&KmeansSchemeFluidX2SB::Recenter, this, image, clusters);
		auto tp2 = tf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), tpb2);
		Guard*       g2 = gs->newGuard("GuardRecenter" + std::to_string(i), { v1_ }, tp2, {});
		//gs->iSync();
		tp0status = new __fluid__<bool>(tp2->isfinished());
		v0 = new ValveEQ<bool>(tp0status, true);
		//rate = rate - i  *1.0 / n;
	}

	auto tp3 = tf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	auto v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	gs->sync(tp3);

	double sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	SyncLogger::print("DIST: ", sum);
	this->sum = sum;

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}
/*
void KmeansSchemeFluidX2pq::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(2);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

	double  rate = 1;
	SyncLogger::print("=====KmeansSchemeFluidX2 thread_num:%d=====", this->thread_num);

	for (int i = 0; i < n; ++i) {
		loopindicator = "loop" + std::to_string(i);
		//int* counter = new int(0);
		//std::vector<int>* counter = new std::vector<int>;
		//counter->push_back(0);
		//counter->insert(counter->begin(), this->thread_num-1, 0);
 		int *call_num1__anything = new int(0);
		auto call_num1 = new __fluid__<int>(call_num1__anything);
		//printf("counter size: %d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",counter->size());
		auto tpb1 = std::bind(&KmeansRandom::AssignCluster2_stable, this, image, clusters, this->thread_num, std::placeholders::_1);
		auto tp1 = tf->newTask<decltype(tpb1), int*>("AssignCluster" + std::to_string(i), tpb1, call_num1->p);
		Guard* g1 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1, {});

		//auto tp1status = new __fluid__<std::vector<int>>(counter);

		//if (i < 10) rate = 1;
		if (rate <= 0.025) rate = 0.025;

		//auto v1 = new ValveGT_vec<int>(tp1status, image->h*image->w*rate);
		auto v0_ = v1.init(call_num1,image->size*rate/this->thread_num);

		//auto tpb2 = std::bind(&KmeansRandom::Recenter2, this, image, clusters, this->thread_num);
		//auto tpb2 = std::bind(&KmeansRandomChangedRecenter::Recenter2, this, image, clusters, this->thread_num);
		auto tpb2 = std::bind(&KmeansRandom::Recenter, this, image, clusters);
		auto tp2 = tf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), tpb2);
		Guard*       g2 = gs->newGuard("GuardRecenter" + std::to_string(i), { v0_ }, tp2, { g1 });
		gs->iSync();
		tp0status = new __fluid__<bool>(tp2->isfinished());
		v0 = new ValveEQ<bool>(tp0status, true);
		rate = rate - i  *1.0 / n;
	}

	auto tp3 = tf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	auto v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	gs->sync(tp3);

	double sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	SyncLogger::print("DIST: ", sum);
	this->sum = sum;

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}*/
/*
void KmeansSchemeFluidX3::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(2);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

	double  rate = 1;
	SyncLogger::print("=====KmeansSchemeFluidX3 thread_num:%d=====", this->thread_num);

	
	__fluid__ image.producer() = std::bind(&KmeansRandom::AssignCluster2, this, image, clusters);
	__fluid__ image.consumer() = std::bind(&KmeansRandomChangedRecenter::Recenter2, this, image, clusters, this->thread_num);
	__fluid__ v0 = new ValveEQ<bool>(, true);
	for(int i = 0; i < n; ++i) {
		__fluid__ image.producer();
		__fluid__ Guard* g1 = newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, image.profucer(), {});
		if (rate <= 0.05) rate = 0.05;
		__fluid__ Valve v1 = new ValveGT_vec<int>(image.producer().CALL_NUM(), image->h*image->w*rate);
		__fluid__ image.consumer();
		__fluid__ Guard* g2 = newGuard("GuardRecenter" + std::to_string(i), { v1 }, image.consumer(), { g1 });
		__fluid__ Valve v0 = new ValveEQ<bool>(image.consumer().isfinished());
		rate = rate - i  *1.0 / n;
	}
	
	for (int i = 0; i < n; ++i) {
		loopindicator = "loop" + std::to_string(i);
		//int* counter = new int(0);
		std::vector<int>* counter = new std::vector<int>;
		counter->push_back(0);
		counter->insert(counter->begin(), this->thread_num-1, 0);
		//printf("counter size: %d!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",counter->size());
		auto tpb1 = std::bind(&KmeansRandom::AssignCluster2, this, image, clusters, std::placeholders::_1);
		auto tp1 = tf->newTask<decltype(tpb1), std::vector<int>*>("AssignCluster" + std::to_string(i), tpb1, counter);
		Guard* g1 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1, {});

		auto tp1status = new __fluid__<std::vector<int>>(counter);

		//if (i < 10) rate = 1;
		if (rate <= 0.05) rate = 0.05;

		auto v1 = new ValveGT_vec<int>(tp1status, image->h*image->w*rate);

		//auto tpb2 = std::bind(&KmeansRandom::Recenter2, this, image, clusters, this->thread_num);
		auto tpb2 = std::bind(&KmeansRandomChangedRecenter::Recenter2, this, image, clusters, this->thread_num);
		//auto tpb2 = std::bind(&KmeansRandom::Recenter, this, image, clusters);
		auto tp2 = tf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), tpb2);
		Guard*       g2 = gs->newGuard("GuardRecenter" + std::to_string(i), { v1 }, tp2, { g1 });
		gs->iSync();
		tp0status = new __fluid__<bool>(tp2->isfinished());
		v0 = new ValveEQ<bool>(tp0status, true);
		rate = rate - i  *1.0 / n;
	}

	auto tp3 = tf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	auto v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	gs->sync(tp3);

	double sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	SyncLogger::print("DIST: ", sum);

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}*/

void KmeansSchemePPL::segmentImage(RgbImage* image, Clusters* clusters, int n) {
	TaskFactory tf;
	AggressiveGS gs(1);
	auto tp0 = tf.newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

	for (int i = 0; i < n; ++i) {
		int* counter = new int(0);
		auto tpb1 = std::bind(&KmeansRandom::AssignCluster, this, image, clusters, counter);
		auto tp1 = tf.newTask<decltype(tpb1)>("AssignCluster" + std::to_string(i), tpb1);
		Guard* g1 = gs.newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1, {});

		auto tp1status = new __fluid__<bool>(tp1->isfinished());
		auto v1 = new ValveEQ<bool>(tp1status, true);
		
		auto tpb2 = std::bind(&KmeansRandom::Recenter, this, image, clusters);
		auto tp2 = tf.newTask<decltype(tpb2)>("Recenter" + std::to_string(i), tpb2);
		Guard*       g2 = gs.newGuard("GuardRecenter" + std::to_string(i), { v1 }, tp2, { g1 });

		tp0status = new __fluid__<bool>(tp2->isfinished());
		v0 = new ValveEQ<bool>(tp0status, true);
		//AssignCluster(image, clusters);
		//Recenter(image, clusters);
	}

	auto tp3 = tf.newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	auto v3 = new ValveEQ<bool>(tp3status, true);
	//Guard*       g3 = new Guard("EndGuard" , { v3 }, tp3, {});
	while (*(tp3->parent->isfinished()) == false)
		; // HARD HACK!

		  //std::this_thread::sleep_for(std::chrono::seconds(20));
		  //delete g1, g2;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}

void KmeansSchemeFluidX_E::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	ttf = new TaskFactory();
	ggs = new AggressiveGS(20);
	auto tp0 = ttf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

	double  rate = 1;
	rate = this->rate;
	SyncLogger::print("=====KmeansSchemeFluidX_E=====");
	//time_tag.resize(n*10);
	double pre = 100000000.0;
	double sum = 0.0;
	Data *d1 = new Data;
	d1->set_data((void*)image);
	double* pre_sum = new double(1000000000.0);
	double* this_sum;
	// n=;
	for (int i = 0; i < n; ++i) {
		std::cout << "loop: " << i << std::endl;
		loopindicator = "loop" + std::to_string(i);

		Data *d2 = new Data;
		d2->set_data((void*)clusters);

		int* counter = new int(0);
		auto tpb1 = std::bind(&KmeansSchemeFluidX_E::AssignCluster, this, image, clusters, std::placeholders::_1);
		auto tp1 = ttf->newTask<decltype(tpb1), int*>("AssignCluster" + std::to_string(i), {d1}, {d2}, tpb1, counter);
		Guard* g1 = ggs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, {}, tp1, {});
		g1->set_root();
		guard_log.push_back(g1);

		d1 = new Data;
		d1->set_data((void*)image);

		auto tp1status = new __fluid__<int>(counter);

		auto v1 = new ValveGT<int>(tp1status, image->h*image->w*rate);

		auto pre_sumstatus = new __fluid__<double>(pre_sum);
		this_sum = new double(0);
		auto this_sumstatus = new __fluid__<double>(this_sum);
		auto v2 = new ValveGT2<double>(pre_sumstatus, this_sumstatus);


		auto tpb2 = std::bind(&KmeansSchemeFluidX_E::Recenter, this, image, clusters, this_sum);
		auto tp2 = ttf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), {d2}, {d1}, tpb2);
		Guard*       g2 = ggs->newGuard("GuardRecenter" + std::to_string(i), { v1 }, {}, tp2, {  });
		g2->set_leaf();
		guard_log.push_back(g2);
		ggs->sync(tp2);

		tp0status = new __fluid__<bool>(tp2->isfinished());
		v0 = new ValveEQ<bool>(tp0status, true);

		pre_sum = this_sum;
	}

	auto tp3 = ttf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	auto v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	ggs->sync(tp3);

	sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	sum = print_sum_to_DIST1(image, clusters);
	SyncLogger::print("DIST: ", sum);
	this->sum = sum;

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}

void KmeansSchemeFluidX_E::Recenter(RgbImage* image, Clusters* clusters, double *this_sum) {
	printf("KmeansSchemeFluidX_E::Recenter\n");

    //struct  timeval    tv;
	//struct  timezone   tz;
	//gettimeofday(&tv,&tz);
	//time_tag.push_back("Recenter start: "+std::to_string(tv.tv_sec)+"\t: "+std::to_string(tv.tv_usec));

	auto r = new double[clusters->k];
	auto g = new double[clusters->k];
	auto b = new double[clusters->k];
	auto n = new int[clusters->k];
	//printf("shi zhege me??????\n");
	for (int c = 0; c < clusters->k; ++c) {
		r[c] = 0.;
		g[c] = 0.;
		b[c] = 0.;
		n[c] = 0;
	}

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			r[image->pixels[y][x].cluster] += image->pixels[y][x].r;
			g[image->pixels[y][x].cluster] += image->pixels[y][x].g;
			b[image->pixels[y][x].cluster] += image->pixels[y][x].b;
			n[image->pixels[y][x].cluster] += 1;
		}
	}
	for (int c = 0; c < clusters->k; ++c) {
		SyncLogger::print(loopindicator, "clusters-NC" + std::to_string(c), " ", n[c]);

		clusters->centroids[c].n = n[c];
	}
	for (int c = 0; c < clusters->k; ++c) {
		if (n[c] == 0) {
			clusters->centroids[c].r = r[c];
			clusters->centroids[c].g = g[c];
			clusters->centroids[c].b = b[c];

		}
		else if (n[c] != 0) {

			clusters->centroids[c].r = r[c] / n[c];
			clusters->centroids[c].g = g[c] / n[c];
			clusters->centroids[c].b = b[c] / n[c];

		}
	}
	for (int c = 0; c < clusters->k; ++c) {
		//SyncLogger::print("clusters: ", c, " ", clusters->centroids[c].n, clusters->centroids[c].r, clusters->centroids[c].g, clusters->centroids[c].b);
		SyncLogger::print(loopindicator, "clusters-FN" + std::to_string(c), " ", n[c], clusters->centroids[c].n, clusters->centroids[c].r, clusters->centroids[c].g, clusters->centroids[c].b);

	}
	double sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	*this_sum = sum;
	delete[] r;
	delete[] g;
	delete[] b;
	delete[] n;
	printf("KmeansSchemeFluidX_E::Recenter End\n");

	//gettimeofday(&tv,&tz);
	//time_tag.push_back("Recenter finish: "+std::to_string(tv.tv_sec)+"\t: "+std::to_string(tv.tv_usec));
	//double sum = print_sum_to_DIST1(image,clusters); //print_sum//
	//SyncLogger::print("DIST: ", sum);
}

void KmeansSchemeFluidX2_E::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	auto tf = new TaskFactory();
	auto gs = new AggressiveGS(32);
	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

	double  rate = 1;
	rate = this->rate;
	SyncLogger::print("=====KmeansSchemeFluidX2_E thread_num:%d=====", this->thread_num);

	Data *d1 = new Data;
	d1->set_data((void*)image);
	double* pre_sum = new double(1000000000.0);
	double* this_sum;
	for (int i = 0; i < n; ++i) {
		loopindicator = "loop" + std::to_string(i);

/*
 		int *call_num1__anything = new int(0);
		auto call_num1 = new __fluid__<int>(call_num1__anything);

		auto tpb1_0 = std::bind(&KmeansRandom::AssignCluster2, this, image, clusters, this->thread_num, 0, std::placeholders::_1);
		auto tp1_0 = tf->newTask<decltype(tpb1_0), int*>("AssignCluster" + std::to_string(i), tpb1_0, call_num1->p);
		Guard* g1_0 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_0, {});

		auto tpb1_1 = std::bind(&KmeansRandom::AssignCluster2, this, image, clusters, this->thread_num, 1, std::placeholders::_1);
		auto tp1_1 = tf->newTask<decltype(tpb1_1), int*>("AssignCluster" + std::to_string(i), tpb1_1, call_num1->p);
		Guard* g1_1 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_1, {});
		
		auto tpb1_2 = std::bind(&KmeansRandom::AssignCluster2, this, image, clusters, this->thread_num, 2, std::placeholders::_1);
		auto tp1_2 = tf->newTask<decltype(tpb1_2), int*>("AssignCluster" + std::to_string(i), tpb1_2, call_num1->p);
		Guard* g1_2 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_2, {});
		
		auto tpb1_3 = std::bind(&KmeansRandom::AssignCluster2, this, image, clusters, this->thread_num, 3, std::placeholders::_1);
		auto tp1_3 = tf->newTask<decltype(tpb1_3), int*>("AssignCluster" + std::to_string(i), tpb1_3, call_num1->p);
		Guard* g1_3 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_3, {});

		auto v0_ = v1.init(call_num1,image->size*rate); */
		//auto tp1status = new __fluid__<std::vector<int>>(counter);

		//if (i < 10) rate = 1;
		//if (rate <= 0.025) rate = 0.025;

		//auto v1 = new ValveGT_vec<int>(tp1status, image->h*image->w*rate);
		std::vector<Data*> d1;
		std::vector<Data*> d2;
		std::set<Data*> d2__;
		for (int j=0; j<this->thread_num; j++) {
			Data *d1_ = new Data;
			d1_->set_data((void*)image);
			d1.push_back(d1_);
			Data *d2_ = new Data;
			d2_->set_data((void*)clusters);
			d2.push_back(d2_);
			d2__.insert(d2_);
		}

		////#pragma call_num{__call_num__ call_num1 this->thread_num}
 		std::vector<int> *call_num1__anything = new std::vector<int>;
 		call_num1__anything->resize(this->thread_num);
		auto call_num1 = new __fluid__<std::vector<int>>(call_num1__anything); //&((*(call_num1->p))[i])

		////<<<g1,{v0},{},call_num1,this->thread_num>>>AssignCluster2(image, clusters, this->thread_num);
		std::set<Guard*> g1;
		for(int th=0; th<this->thread_num; th++)
		{
			auto tpb1_0 = std::bind(&KmeansSchemeFluidX2_E::AssignCluster2, this, image, clusters, this->thread_num, th, std::placeholders::_1);
			auto tp1_0 = tf->newTask<decltype(tpb1_0), std::vector<int>*>("AssignCluster" + std::to_string(i), {d1[th]}, {d2[th]}, tpb1_0, call_num1->p);
			Guard* g1_0 = gs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, tp1_0, {});
			g1_0->set_root();
			g1.insert(g1_0);
		}
		//d1 = new Data;
		//d1->set_data((void*)image);

		auto v0_ = v2.init(call_num1, image->h*image->w*rate);

		auto pre_sumstatus = new __fluid__<double>(pre_sum);
		this_sum = new double(0);
		auto this_sumstatus = new __fluid__<double>(this_sum);
		auto v2 = new ValveGT2<double>(pre_sumstatus, this_sumstatus);

		Data *d3 =new Data;
		d3->set_data((void*)image);
		//auto tpb2 = std::bind(&KmeansRandom::Recenter2, this, image, clusters, this->thread_num);
		//auto tpb2 = std::bind(&KmeansRandomChangedRecenter::Recenter2, this, image, clusters, this->thread_num);
		auto tpb2 = std::bind(&KmeansSchemeFluidX2_E::Recenter, this, image, clusters, this_sum);
		auto tp2 = tf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), d2__, {d3}, tpb2);
		Guard*       g2 = gs->newGuard("GuardRecenter" + std::to_string(i), { v0_ }, {v2}, tp2, {  });
		//Guard*       g2 = gs->newGuard("GuardRecenter" + std::to_string(i), { v0_ }, {v2}, tp2, g1);
		g2->set_leaf();
		//gs->iSync();
		tp0status = new __fluid__<bool>(tp2->isfinished());
		v0 = new ValveEQ<bool>(tp0status, true);
		//rate = rate - i  *1.0 / n;
		pre_sum = this_sum;
	}

	//auto tp3 = tf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	//auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	//auto v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	gs->sync(tp0);

	double sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	SyncLogger::print("DIST: ", sum);
	this->sum = sum;

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}


void KmeansSchemeFluidX22_E::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	auto ttf = new TaskFactory();
	auto ggs = new AggressiveGS(24);
	auto tp0 = ttf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

	double  rate = 1;
	rate = this->rate;
	SyncLogger::print("=====KmeansSchemeFluidX2_E thread_num:%d=====", this->thread_num);

	Data *d1 = new Data;
	d1->set_data((void*)image);
	double* pre_sum = new double(1000000000.0);
	double* this_sum;
	for (int i = 0; i < n; ++i) {
		std::cout << "loop: " << i << std::endl;
		loopindicator = "loop" + std::to_string(i);

		Data *d2 = new Data;
		d2->set_data((void*)clusters);

 		std::vector<int> *counter = new std::vector<int>;
 		counter->resize(this->thread_num);

		auto tpb1 = std::bind(&KmeansSchemeFluidX22_E::AssignCluster22, this, image, clusters, this->thread_num, std::placeholders::_1);
		auto tp1 = ttf->newTask<decltype(tpb1), std::vector<int>*>("AssignCluster" + std::to_string(i), {d1}, {d2}, tpb1, counter);
		Guard* g1 = ggs->newGuard("GuardAssignCluster" + std::to_string(i), { v0 }, {}, tp1, {});
		g1->set_root();
		//guard_log.push_back(g1);

		d1 = new Data;
		d1->set_data((void*)image);

		//auto tp1status = new __fluid__<int>(counter);
		auto call_num1 = new __fluid__<std::vector<int>>(counter);

		//auto v1 = new ValveGT<int>(tp1status, image->h*image->w*rate);
		auto v1 = new ValveGT_vec<int>(call_num1, image->h*image->w*rate);

		auto pre_sumstatus = new __fluid__<double>(pre_sum);
		this_sum = new double(0);
		auto this_sumstatus = new __fluid__<double>(this_sum);
		auto v2 = new ValveGT2<double>(pre_sumstatus, this_sumstatus);


		auto tpb2 = std::bind(&KmeansSchemeFluidX22_E::Recenter, this, image, clusters, this_sum);
		auto tp2 = ttf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), {d2}, {d1}, tpb2);
		Guard*       g2 = ggs->newGuard("GuardRecenter" + std::to_string(i), { v1 }, {}, tp2, {  });
		g2->set_leaf();
		//guard_log.push_back(g2);
		ggs->sync(tp2);

		tp0status = new __fluid__<bool>(tp2->isfinished());
		v0 = new ValveEQ<bool>(tp0status, true);

		pre_sum = this_sum;
	}

	//auto tp3 = tf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	//auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	//auto v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	ggs->sync(tp0);

	double sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	SyncLogger::print("DIST: ", sum);
	this->sum = sum;

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}

void KmeansSchemeFluidXS_E::segmentImage(RgbImage* image, Clusters* clusters, int n) {

	ttf = new TaskFactory();
	ggs = new AggressiveGS(20);
	auto tp0 = ttf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
	auto tp0status = new __fluid__<bool>(tp0->isfinished());
	auto v0 = new ValveEQ<bool>(tp0status, true);

	tp0->run();

	double  rate = 1;
	rate = this->rate;
	SyncLogger::print("=====KmeansSchemeFluidXS_E=====");
	//time_tag.resize(n*10);
	double pre = 100000000.0;
	double sum = 0.0;
	Data *d1 = new Data;
	d1->set_data((void*)image);
	double* pre_sum = new double(1000000000.0);
	double* this_sum;
	// n=;
	for (int i = 0; i < n; ++i) {
		std::cout << "loop: " << i << std::endl;
		loopindicator = "loop" + std::to_string(i);

		Data *d2 = new Data;
		d2->set_data((void*)clusters);

		int* counter = new int(0);
		auto tpb1 = std::bind(&KmeansSchemeFluidXS_E::AssignClusterXS_E, this, image, clusters, std::placeholders::_1);
		auto tp1 = ttf->newTask<decltype(tpb1), int*>("AssignClusterXS_E" + std::to_string(i), {d1}, {d2}, tpb1, counter);
		Guard* g1 = ggs->newGuard("GuardAssignClusterXS_E" + std::to_string(i), {  }, {}, tp1, {});
		g1->set_root();
		guard_log.push_back(g1);

		d1 = new Data;
		d1->set_data((void*)image);

		auto tp1status = new __fluid__<int>(counter);

		auto v1 = new ValveGT<int>(tp1status, image->h*image->w*rate);

		auto pre_sumstatus = new __fluid__<double>(pre_sum);
		this_sum = new double(0);
		auto this_sumstatus = new __fluid__<double>(this_sum);
		auto v2 = new ValveGT2<double>(pre_sumstatus, this_sumstatus);


		auto tpb2 = std::bind(&KmeansSchemeFluidXS_E::Recenter, this, image, clusters, this_sum);
		auto tp2 = ttf->newTask<decltype(tpb2)>("Recenter" + std::to_string(i), {d2}, {d1}, tpb2);
		Guard*       g2 = ggs->newGuard("GuardRecenter" + std::to_string(i), { v1 }, {}, tp2, {  });
		g2->set_leaf();
		guard_log.push_back(g2);
		ggs->synctask(tp2);

		tp0status = new __fluid__<bool>(tp2->isfinished());
		v0 = new ValveEQ<bool>(tp0status, true);

		pre_sum = this_sum;
	}

	auto tp3 = ttf->newTask<void(*)(void *), void *>("End", &nullfun, NULL);
	auto tp3status = new __fluid__<bool>(tp3->parent->isfinished());
	auto v3 = new ValveEQ<bool>(tp3status, true);
	//while (*(tp3->parent->isfinished()) == false)
	//	; // HARD HACK!
	ggs->sync(tp3);

	sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	sum = print_sum_to_DIST1(image, clusters);
	SyncLogger::print("DIST: ", sum);
	this->sum = sum;

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			image->pixels[y][x].r = clusters->centroids[image->pixels[y][x].cluster].r;
			image->pixels[y][x].g = clusters->centroids[image->pixels[y][x].cluster].g;
			image->pixels[y][x].b = clusters->centroids[image->pixels[y][x].cluster].b;
		}
	}
}

void KmeansSchemeFluidXS_E::Recenter(RgbImage* image, Clusters* clusters, double *this_sum) {
	printf("KmeansSchemeFluidXS_E::Recenter\n");

    //struct  timeval    tv;
	//struct  timezone   tz;
	//gettimeofday(&tv,&tz);
	//time_tag.push_back("Recenter start: "+std::to_string(tv.tv_sec)+"\t: "+std::to_string(tv.tv_usec));

	auto r = new double[clusters->k];
	auto g = new double[clusters->k];
	auto b = new double[clusters->k];
	auto n = new int[clusters->k];
	//printf("shi zhege me??????\n");
	for (int c = 0; c < clusters->k; ++c) {
		r[c] = 0.;
		g[c] = 0.;
		b[c] = 0.;
		n[c] = 0;
	}

	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			r[image->pixels[y][x].cluster] += image->pixels[y][x].r;
			g[image->pixels[y][x].cluster] += image->pixels[y][x].g;
			b[image->pixels[y][x].cluster] += image->pixels[y][x].b;
			n[image->pixels[y][x].cluster] += 1;
		}
	}
	for (int c = 0; c < clusters->k; ++c) {
		SyncLogger::print(loopindicator, "clusters-NC" + std::to_string(c), " ", n[c]);

		clusters->centroids[c].n = n[c];
	}
	for (int c = 0; c < clusters->k; ++c) {
		if (n[c] == 0) {
			clusters->centroids[c].r = r[c];
			clusters->centroids[c].g = g[c];
			clusters->centroids[c].b = b[c];

		}
		else if (n[c] != 0) {

			clusters->centroids[c].r = r[c] / n[c];
			clusters->centroids[c].g = g[c] / n[c];
			clusters->centroids[c].b = b[c] / n[c];

		}
	}
	for (int c = 0; c < clusters->k; ++c) {
		//SyncLogger::print("clusters: ", c, " ", clusters->centroids[c].n, clusters->centroids[c].r, clusters->centroids[c].g, clusters->centroids[c].b);
		SyncLogger::print(loopindicator, "clusters-FN" + std::to_string(c), " ", n[c], clusters->centroids[c].n, clusters->centroids[c].r, clusters->centroids[c].g, clusters->centroids[c].b);

	}
	double sum = 0;
	for (int y = 0; y < image->h; y++) {
		for (int x = 0; x < image->w; x++) {
			sum += (pow(image->pixels[y][x].r - clusters->centroids[image->pixels[y][x].cluster].r, 2) +
				pow(image->pixels[y][x].g - clusters->centroids[image->pixels[y][x].cluster].g, 2) +
				pow(image->pixels[y][x].b - clusters->centroids[image->pixels[y][x].cluster].b, 2));
		}
	}
	*this_sum = sum;
	delete[] r;
	delete[] g;
	delete[] b;
	delete[] n;
	printf("KmeansSchemeFluidXS_E::Recenter End\n");

	//gettimeofday(&tv,&tz);
	//time_tag.push_back("Recenter finish: "+std::to_string(tv.tv_sec)+"\t: "+std::to_string(tv.tv_usec));
	//double sum = print_sum_to_DIST1(image,clusters); //print_sum//
	//SyncLogger::print("DIST: ", sum);
}
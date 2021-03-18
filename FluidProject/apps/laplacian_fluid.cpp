#include "laplacian.h"
#include "laplacian_fluid.h"

using namespace example;

void LaplacianFluid_E::segmentImage(tifImage *image)
{
  std::cout << "rate: " << rate << std::endl;

  auto tf = new TaskFactory();
  auto gs = new AggressiveGS(20);
  auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
  auto tp0status = new __fluid__<bool>(tp0->isfinished());
  auto v0 = new ValveEQ<bool>(tp0status, true);
  tp0->run();

  Guard *g1;
  Guard *g2;

  ////#pragma Data *d1(image); 
  ////#pragma Data *d2(image); 
  ////#pragma Data *d3(image); 
  Data *d1 = new Data;
  d1->set_data((void*)image);
  Data *d2 = new Data;
  d2->set_data((void*)image);
  Data *d3 = new Data;
  d3->set_data((void*)image);

  ////#pragma call_num {__call_num__ call_num1;} // buyao

  ////#pragma call_num call_num1;
  int *call_num1__anything = new int(0);
  auto call_num1 = new __fluid__<int>(call_num1__anything);

  if (mode == 1 || mode == 2) {
    ////#pragma task <<<g1,{},{},{d1},{d2}>>>Gaussian_filter(image,call_num);
    auto tpb__g1 = std::bind(&LaplacianFluid_E::Gaussian_filter, this, image, std::placeholders::_1);
    auto tp__g1 = tf->newTask<decltype(tpb__g1), int*>("Gaussian_filter", {d1}, {d2}, tpb__g1, call_num1->p);
    g1 = gs->newGuard("Gaussian_filter", {}, {}, tp__g1, {});
  } else if (mode == 3 || mode == 4) {
    ////#pragma task <<<g1,{},{},{d1},{d2}>>>Mean_filter(image,call_num);
    auto tpb__g1 = std::bind(&LaplacianFluid_E::Mean_filter, this, image, std::placeholders::_1);
    auto tp__g1 = tf->newTask<decltype(tpb__g1), int*>("Mean_filter", {d1}, {d2}, tpb__g1, call_num1->p);
    g1 = gs->newGuard("Mean_filter", {}, {}, tp__g1, {});
  }


  ////g1->set_root;
  g1->set_root();
  guard_log.push_back(g1);

  ////<<<g2,{v1(call_num1,image->size*rate)},{g1},call_num2>>>Sobel_filter(image); // buyao

  ////#pragma valve v0_ = v1.init(call_num1,image->size*rate);
  ////#pragma valve v1_ = v1.init(call_num1,image->size+1);
  auto v0_ = v1.init(call_num1,image->size*rate);
  auto v1_ = v1.init(call_num1, 0.8*(image->h-10)*(image->w-10));


  if (mode == 1 || mode == 3) {
    //Sobel_filter(image);
    ////#pragma task <<<g2,{v0_},{v1_},{d2},{d3}>>Sobel_filter(image);
    auto tpb__g2 = std::bind(&LaplacianFluid_E::Sobel_filter, this, image);
    auto tp__g2 = tf->newTask<decltype(tpb__g2)>("Sobel_filter", {d2}, {d3}, tpb__g2);
    g2 = gs->newGuard("Sobel_filter", {v0_}, {v1_}, tp__g2, {});
  } else if (mode == 2 || mode == 4) {
    //Laplacian_filter(image);
    ////#pragma task <<<g2,{v0_},{v1_},{d2},{d3}>>Laplacian_filter(image);
    auto tpb__g2 = std::bind(&LaplacianFluid_E::Laplacian_filter, this, image);
    auto tp__g2 = tf->newTask<decltype(tpb__g2)>("Laplacian_filter", {d2}, {d3}, tpb__g2);
    g2 = gs->newGuard("Laplacian_filter", {v0_}, {v1_}, tp__g2, {});
  }

  /////g2->set_leaf();
  g2->set_leaf();
  guard_log.push_back(g2);
  ////synctask(g2);
  gs->sync(g2->task);

}



////void SobelFluid::Gaussian_filter(tifImage *image)
void LaplacianFluid_E::Gaussian_filter(tifImage *image, int *call_num)
{
  std::cout << "Gaussian_filter start!!!\n";
  double sigma_2 = sigma * sigma;
  double sum = 0.0;
  double *kernel = new double[121];

  for(int i=0;i<11;i++) {
    for(int j=0;j<11;j++) {
        kernel[11*i+j] = exp((-(i - 5) * (i - 5) - (j - 5) * (j - 5)) / (2 * sigma_2)) / (2 * 3.1415 * sigma_2);
        sum = sum + kernel[11*i+j];
    }
  }
  for(int i=0; i<image->h; i++) {
    for(int j=0; j<image->w; j++) {
      image->gaussian[i][j] = image->pixels[i][j];
    }
  }
  for(int i=0;i<121;i++)
    kernel[i] = kernel[i] / sum;

  for(int i=5; i<image->h-5; i++) {
    for(int j=5; j<image->w-5; j++) {
      ////#pragma call_num {conv_gaussian(image->pixels, kernel, i, j, image->gaussian);}
      {conv_gaussian(image->pixels, kernel, i, j, image->gaussian);
      //(*call_num)++;
      }
      //std::cout << "call_num: " << *call_num << std::endl;
    }
    (*call_num)+=image->w - 9;
  }
  std::cout << "Gaussian_filter end!!!\n";
  //printf("Gaussian_filter end %d!!!\n", *call_num);
}


////void SobelFluid::Mean_filter(tifImage *image)
void LaplacianFluid_E::Mean_filter(tifImage *image, int *call_num)
{
  std::cout << "Mean_filter start!!!\n";
  double sigma_2 = sigma * sigma;
  double sum = 0.0;
  int kernel_size = 11;
  double *kernel = new double[kernel_size * kernel_size];

  for(int i=0;i<kernel_size;i++) {
    for(int j=0;j<kernel_size;j++) {
        kernel[kernel_size*i+j] = 1.0 / (kernel_size * kernel_size);
    }
  }
  for(int i=0; i<image->h; i++) {
    for(int j=0; j<image->w; j++) {
      image->gaussian[i][j] = image->pixels[i][j];
    }
  }

  for(int i=(kernel_size / 2); i<image->h-(kernel_size / 2); i++) {
    for(int j=(kernel_size / 2); j<image->w-(kernel_size / 2); j++) {
      ////#pragma call_num {conv_gaussian(image->pixels, kernel, i, j, image->gaussian);}
      {conv_mean(image->pixels, kernel, kernel_size, i, j, image->gaussian);
      //(*call_num)++;
      }
      //std::cout << "call_num: " << *call_num << std::endl;
    }
    (*call_num)+=image->w - 9;
  }
  std::cout << "Mean_filter end!!!\n";
}


////void SobelFluid::Sobel_filter(tifImage *image)
void LaplacianFluid_E::Sobel_filter(tifImage *image)
{
  std::cout << "Sobel_filter start!!!\n";
  double *kernel_x = new double[9];
  double *kernel_y = new double[9];
  kernel_x[0]=-1.0;
  kernel_x[1]=0.0;
  kernel_x[2]=1.0;
  kernel_x[3]=-2.0;
  kernel_x[4]=0.0;
  kernel_x[5]=2.0;
  kernel_x[6]=-1.0;
  kernel_x[7]=0.0;
  kernel_x[8]=1.0;
  kernel_y[0]=-1.0;
  kernel_y[1]=-2.0;
  kernel_y[2]=-1.0;
  kernel_y[3]=0.0;
  kernel_y[4]=0.0;
  kernel_y[5]=0.0;
  kernel_y[6]=1.0;
  kernel_y[7]=2.0;
  kernel_y[8]=1.0;

  for(int x=1; x<image->h-1 ;x++) {
    for(int y=1; y<image->w-1 ;y++) {
      ////#pragma call_num {conv_sobel(image->gaussian, kernel_x, kernel_y, x, y, image->sobel);}
      conv_sobel(image->gaussian, kernel_x, kernel_y, x, y, image->sobel);
      //(*call_num)++;}
    }
  }
  std::cout << "Sobel_filter end!!!\n";
}

////void SobelFluid::Sobel_filter(tifImage *image)
void LaplacianFluid_E::Laplacian_filter(tifImage *image)
{
  std::cout << "Laplacian_filter start!!!\n";
  double *kernel_x = new double[9];
  kernel_x[0]=-1.0;
  kernel_x[1]=-1.0;
  kernel_x[2]=-1.0;
  kernel_x[3]=-1.0;
  kernel_x[4]=8.0;
  kernel_x[5]=-1.0;
  kernel_x[6]=-1.0;
  kernel_x[7]=-1.0;
  kernel_x[8]=-1.0;

  for(int x=1; x<image->h-1 ;x++) {
    for(int y=1; y<image->w-1 ;y++) {
      ////#pragma call_num {conv_sobel(image->gaussian, kernel_x, kernel_y, x, y, image->sobel);}
      conv_laplacian(image->gaussian, kernel_x, x, y, image->sobel);
      //(*call_num)++;}
    }
  }
  std::cout << "Laplacian_filter end!!!\n";
}

//////////////////////////////////////////////////////////////Multi-threads/////////////////////////////////////

void LaplacianFluid_multi_E::segmentImage(tifImage *image)
{
  std::cout << "rate: " << rate << std::endl;

  auto tf = new TaskFactory();
  auto gs = new AggressiveGS(64);
  auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
  auto tp0status = new __fluid__<bool>(tp0->isfinished());
  auto v0 = new ValveEQ<bool>(tp0status, true);
  tp0->run();

  ////#pragma Data *d1(image); 
  ////#pragma Data *d2(image); 
  ////#pragma Data *d3(image); 
  /*
    Data *d1 = new Data;
    d1->set_data((void*)image);
    Data *d2 = new Data;
    d2->set_data((void*)image);
    Data *d3 = new Data;
    d3->set_data((void*)image);
    */
    Data **d1 = new Data*[this->thread_num];
    //d1->set_data((void*)image);
    Data **d2 = new Data*[this->thread_num];
    //d2->set_data((void*)image);
    Data **d3 = new Data*[this->thread_num];
    //d3->set_data((void*)image);
    __fluid__<int>** call_num1 = new __fluid__<int>*[this->thread_num];
  std::set<Guard*> gg11;
  for (int th=0; th<1; th++) {
  //for (int th=0; th<this->thread_num; th++) {
    d1[th] = new Data;
    d1[th]->set_data((void*)image);
    d2[th] = new Data;
    d2[th]->set_data((void*)image);
    d3[th] = new Data;
    d3[th]->set_data((void*)image);
    ////#pragma call_num {__call_num__ call_num1;} // buyao

    ////#pragma call_num call_num1;
    int *call_num1__anything = new int(0);
    call_num1[th] = new __fluid__<int>(call_num1__anything);

    //std::vector<int> *call_num2__anything = new std::vector<int>;
    //call_num2__anything->resize(this->thread_num);
    //auto call_num2 = new __fluid__<std::vector<int>>(call_num2__anything); //&((*(call_num1->p))[i])
    ////<<<g1,{},{},call_num1>>>Gaussian_filter(image); // buyao

    ////<<<g1,{},{},{d1},{d2}>>>Gaussian_filter(image,call_num);
    auto tpb__g1 = std::bind(&LaplacianFluid_multi_E::Gaussian_filter_threads, this, this, image, this->thread_num, th, std::placeholders::_1);
    auto tp__g1 = tf->newTask<decltype(tpb__g1), int*>("Gaussian_filter", {d1[th]}, {d2[th]}, tpb__g1, call_num1[th]->p);
    Guard* g1 = gs->newGuard("Gaussian_filter", {}, {}, tp__g1, {});
    ////g1->set_root;
    g1->set_root();
    gg11.insert(g1);
  }

  GuardTask** t_list = new GuardTask*[this->thread_num];
  for (int th=0; th<1; th++) {
  //for (int th=0; th<this->thread_num; th++) {
    ////<<<g2,{v1(call_num1,image->size*rate)},{g1},call_num2>>>Sobel_filter(image); // buyao

    ////#pragma valve v0_ = v1.init(call_num1,image->size*rate);
    ////#pragma valve v1_ = v1.init(call_num1,image->size+1);
    auto v0_ = v1.init(call_num1[th],image->size*rate/this->thread_num);
    auto v1_ = v1.init(call_num1[th],(image->h-12)*(image->w-10)/this->thread_num);
    ////<<<g2,{v0_},{v1_},{d2},{d3}>>Sobel_filter(image);
    auto tpb__g2 = std::bind(&LaplacianFluid_multi_E::Sobel_filter_threads, this, this, image, this->thread_num, th);
    auto tp__g2 = tf->newTask<decltype(tpb__g2)>("Sobel_filter", {d2[th]}, {d3[th]}, tpb__g2);
    Guard* g2 = gs->newGuard("Sobel_filter", {v0_}, {v1_}, tp__g2, {});
    //Guard* g2 = gs->newGuard("Sobel_filter", {v0_}, {v1_}, tp__g2, gg11);
    /////g2->set_leaf();
    g2->set_leaf();
    t_list[th]=tp__g2;
  }
  //Sobel_filter(image); 
  //gs->synctask(tp__g2);
  gs->sync(t_list[0]);
}

////void SobelFluid::Gaussian_filter(tifImage *image)
void LaplacianFluid_multi_E::Gaussian_filter_threads(LaplacianFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id, int *call_num)
{
  std::cout << "Gaussian_filter start!!!\n";

  double *kernel = sb->kernel;
  for(int i=thread_id; i<image->h; i+=num_thread) {
    for(int j=0; j<image->w; j++) {
      image->gaussian[i][j] = image->pixels[i][j];
    }
  }

  int start_i = ((5 - 1) / num_thread) * num_thread + thread_id;
  start_i = start_i >=5 ? start_i : start_i + num_thread;
  for(int i=start_i; i<image->h-5; i+=num_thread) {
    for(int j=5; j<image->w-5; j++) {
      ////#pragma call_num {conv_gaussian(image->pixels, kernel, i, j, image->gaussian);}
      {
        sb->conv_gaussian(image->pixels, kernel, i, j, image->gaussian);
        //(*call_num)++;
      }
      //std::cout << "thread: " << thread_id << ", call_num: " << *call_num << std::endl;
    }
    (*call_num)+=image->w - 9;
  }
  //std::cout << "Gaussian_filter end!!!\n";
  //printf("Gaussian_filter end %d!!!\n", *call_num);
}


////void SobelFluid::Gaussian_filter(tifImage *image)
void LaplacianFluid_multi_E::Gaussian_filter(tifImage *image, int num_thread, std::vector<int> *call_num)
{
  double *kernel = this->kernel;
  std::thread* threads = new std::thread[num_thread-1];
  for(int i=1; i<num_thread; i++) {
    //threads[i-1] = std::thread(Gaussian_filter_threads, this, image, kernel, num_thread, i, call_num);
  }

  for(int i=0; i<image->h; i+=num_thread) {
    for(int j=0; j<image->w; j++) {
      image->gaussian[i][j] = image->pixels[i][j];
    }
  }
  int start_i = ((5-1) / num_thread) * num_thread + num_thread;
  for(int i=start_i; i<image->h-5; i+=num_thread) {
    for(int j=5; j<image->w-5; j++) {
      ////#pragma call_num {conv_gaussian(image->pixels, kernel, i, j, image->gaussian);}
      {conv_gaussian(image->pixels, kernel, i, j, image->gaussian);
      (*call_num)[0]++;}
      //std::cout << "thread: 0, " << "call_num: " << *call_num << std::endl;
    }
  }
  for(int i=1; i<num_thread; i++) {
    //threads[i-1].join();
  }
}



////void SobelFluid::Sobel_filter(tifImage *image)
void LaplacianFluid_multi_E::Sobel_filter_threads(LaplacianFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id)
{
  std::cout << "Sobel_filter start!!!\n";

  double *kernel_x = sb->kernel_x;
  double *kernel_y = sb->kernel_y;
  int start_x = thread_id == 0 ? num_thread : thread_id;
  for(int x=start_x; x<image->h-1 ;x+=num_thread) {
    for(int y=1; y<image->w-1 ;y++) {
      ////#pragma call_num {conv_sobel(image->gaussian, kernel_x, kernel_y, x, y, image->sobel);}
      sb->conv_sobel(image->gaussian, kernel_x, kernel_y, x, y, image->sobel);
      //(*call_num)++;}
    }
  }
}

////void SobelFluid::Sobel_filter(tifImage *image)
void LaplacianFluid_multi_E::Sobel_filter(tifImage *image, int num_thread)
{
  std::thread* threads = new std::thread[num_thread-1];
  for(int i=1; i<num_thread; i++) {
    //threads[i-1] = std::thread(Sobel_filter_threads, this, image, num_thread, i);
  }

  double *kernel_x = this->kernel_x;
  double *kernel_y = this->kernel_y;

  for(int x=num_thread; x<image->h-1; x+=num_thread) {
    for(int y=1; y<image->w-1 ;y++) {
      ////#pragma call_num {conv_sobel(image->gaussian, kernel_x, kernel_y, x, y, image->sobel);}
      conv_sobel(image->gaussian, kernel_x, kernel_y, x, y, image->sobel);
      //(*call_num)++;}
    }
  }
  for(int i=1; i<num_thread; i++) {
    //threads[i-1].join();
  }

}

void Gaussian_filter_threads22(LaplacianFluid_multi_E *sb, tifImage *image, double *kernel, int num_thread, int thread_id) {
  for(int i=thread_id; i<image->h; i+=num_thread) {
    for(int j=0; j<image->w; j++) {
      image->gaussian[i][j] = image->pixels[i][j];
    }
  }

  int start_i = ((5 - 1) / num_thread) * num_thread + thread_id;
  start_i = start_i >=5 ? start_i : start_i + num_thread;
  for(int i=start_i; i<image->h-5; i+=num_thread) {
    for(int j=5; j<image->w-5; j++) {
      {
        sb->conv_gaussian(image->pixels, kernel, i, j, image->gaussian);
      }
    }
  }
}


void LaplacianFluid_multi_E::Gaussian_filter_threads2(LaplacianFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id, int *call_num)
{
  std::cout << "Gaussian_filter start!!!\n";

  double *kernel = sb->kernel;
  std::thread* threads = new std::thread[num_thread-1];
  for(int i=1; i<num_thread; i++) {
    threads[i-1] = std::thread(Gaussian_filter_threads22, this, image, kernel, num_thread, i);
  }

  for(int i=thread_id; i<image->h; i+=num_thread) {
    for(int j=0; j<image->w; j++) {
      image->gaussian[i][j] = image->pixels[i][j];
    }
  }

  int start_i = ((5 - 1) / num_thread) * num_thread + thread_id;
  start_i = start_i >=5 ? start_i : start_i + num_thread;
  for(int i=start_i; i<image->h-5; i+=num_thread) {
    for(int j=5; j<image->w-5; j++) {
      {
        sb->conv_gaussian(image->pixels, kernel, i, j, image->gaussian);
      }
    }
    (*call_num)+=image->w - 9;
  }
}

void Sobel_filter_threads22(LaplacianFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id) {
  double *kernel_x = sb->kernel_x;
  double *kernel_y = sb->kernel_y;
  int start_x = thread_id == 0 ? num_thread : thread_id;
  for(int x=start_x; x<image->h-1 ;x+=num_thread) {
    for(int y=1; y<image->w-1 ;y++) {
      sb->conv_sobel(image->gaussian, kernel_x, kernel_y, x, y, image->sobel);
    }
  }
}

void LaplacianFluid_multi_E::Sobel_filter_threads2(LaplacianFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id)
{
  std::thread* threads = new std::thread[num_thread-1];
  for(int i=1; i<num_thread; i++) {
    threads[i-1] = std::thread(Sobel_filter_threads22, this, image, num_thread, i);
  }
  double *kernel_x = sb->kernel_x;
  double *kernel_y = sb->kernel_y;
  int start_x = thread_id == 0 ? num_thread : thread_id;
  for(int x=start_x; x<image->h-1 ;x+=num_thread) {
    for(int y=1; y<image->w-1 ;y++) {
      sb->conv_sobel(image->gaussian, kernel_x, kernel_y, x, y, image->sobel);
    }
  }
}
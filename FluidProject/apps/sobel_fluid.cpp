#include "sobel.h"
#include "sobel_fluid.h"

using namespace example;


void SobelFluid::segmentImage(tifImage *image)
{
	std::cout << "rate: " << rate << std::endl;

 	auto tf = new TaskFactory();
 	auto gs = new AggressiveGS(2);
 	auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
 	auto tp0status = new __fluid__<bool>(tp0->isfinished());
 	auto v0 = new ValveEQ<bool>(tp0status, true);
 	tp0->run();

 	////#pragma call_num {__call_num__ call_num1;}
 	int *call_num1__anything = new int(0);
	auto call_num1 = new __fluid__<int>(call_num1__anything);
 	////<<<g1,{},{},call_num1>>>Gaussian_filter(image);
	auto tpb__g1 = std::bind(&SobelFluid::Gaussian_filter, this, image, std::placeholders::_1);
	auto tp__g1 = tf->newTask<decltype(tpb__g1), int*>("Gaussian_filter", tpb__g1, call_num1->p);
	Guard* g1 = gs->newGuard("Gaussian_filter", {}, tp__g1, {});

 	////<<<g2,{v1(call_num1,image->size*rate)},{g1},call_num2>>>Sobel_filter(image);
	auto v0_ = v1.init(call_num1,image->size*rate);
	auto tpb__g2 = std::bind(&SobelFluid::Sobel_filter, this, image);
	auto tp__g2 = tf->newTask<decltype(tpb__g2)>("Sobel_filter", tpb__g2);
	Guard* g2 = gs->newGuard("Sobel_filter", {v0_}, tp__g2, {});
	//Sobel_filter(image); 
	gs->synctask(tp__g2);

}


////void SobelFluid::Gaussian_filter(tifImage *image)
void SobelFluid::Gaussian_filter(tifImage *image, int *call_num)
{
  std::cout << "Gaussian_filter start!!!\n";
  double sigma_2 = sigma * sigma;
  double sum = 0.0;
  double *kernel = new double[121];

  for(int i=0; i<image->h; i++) {
    for(int j=0; j<image->w; j++) {
      image->gaussian[i][j] = image->pixels[i][j];
    }
  }
  for(int i=0;i<11;i++) {
    for(int j=0;j<11;j++) {
        kernel[11*i+j] = exp((-(i - 5) * (i - 5) - (j - 5) * (j - 5)) / (2 * sigma_2)) / (2 * 3.1415 * sigma_2);
        sum = sum + kernel[11*i+j];
    }
  }
  for(int i=0;i<121;i++)
    kernel[i] = kernel[i] / sum;

  for(int i=5; i<image->h-5; i++) {
    for(int j=5; j<image->w-5; j++) {
      ////#pragma call_num {conv_gaussian(image->pixels, kernel, i, j, image->gaussian);}
      {conv_gaussian(image->pixels, kernel, i, j, image->gaussian);
      (*call_num)++;}
      //std::cout << "call_num: " << *call_num << std::endl;
    }
  }
  std::cout << "Gaussian_filter end!!!\n";
  //printf("Gaussian_filter end %d!!!\n", *call_num);
}

////void SobelFluid::Sobel_filter(tifImage *image)
void SobelFluid::Sobel_filter(tifImage *image)
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
  //printf("Sobel_filter end!!!\n");
  //printf("Sobel_filter end!!!\n");
}

void SobelFluid::Link_edge(tifImage *image)
{
  int **raw = image->sobel;
  int **raw2 = image->link;

  for(int x=0; x<3; x++)
    for(int y=0; y<image->w; y++)
      if(raw[x][y]==1) raw2[x][y] = 1;
  for(int x=3; x<image->h; x++)
    for(int y=0; y<3; y++)
      if(raw[x][y]==1) raw2[x][y] = 1;
  for(int x=image->h-6; x<image->h; x++)
    for(int y=3; y<image->w; y++)
      if(raw[x][y]==1) raw2[x][y] = 1;
  for(int x=3; x<image->h-6; x++)
    for(int y=image->w-6; y<image->w; y++)
      if(raw[x][y]==1) raw2[x][y] = 1;

  for(int x=3; x<image->h-6; x++) {
    for(int y=3; y<image->w-6; y++) {
      if(raw[x][y]==1) raw2[x][y] = 1;
      if(raw[x][y]==1 && raw[x+3][y+3]==1) {
        raw2[x+1][y+1]=1;
        raw2[x+2][y+2]=1;
      }
      if(raw[x+1][y]==1 && raw[x+2][y+3]==1) {
        raw2[x+1][y+1]=1;
        raw2[x+2][y+2]=1;
      }
      if(raw[x+2][y]==1 && raw[x+1][y+3]==1) {
        raw2[x+1][y+2]=1;
        raw2[x+2][y+1]=1;
      }
      if(raw[x+3][y]==1 && raw[x][y+3]==1) {
        raw2[x+1][y+2]=1;
        raw2[x+2][y+1]=1;
      }
      if(raw[x][y+1]==1 && raw[x+3][y+2]==1) {
        raw2[x+1][y+1]=1;
        raw2[x+2][y+2]=1;
      }
      if(raw[x][y+2]==1 && raw[x+3][y+1]==1) {
        raw2[x+1][y+2]=1;
        raw2[x+2][y+1]=1;
      }
    } // for(int y=3; y<image->w-3; y++)
  } // for(int x=3; x<image->h-3; x++)


}

void SobelFluid::Segment(tifImage *image)
{
  Link_edge(image);
  int **ga = image->segment;
  for(int i=0; i<image->h; i++)
    memset(ga[i],0,sizeof(int)*image->w);

  int **raw = image->link;

  std::cout << raw[518][10] << ' ' << image->size << std::endl;

  int ql=image->size*2;
  int *queue = new int[ql];
  int l=0;
  int r=0;

  std::cout << "OK!\n";
  int i = image->h-10;
  int j = 10;
  while(raw[i][j] == 1)
  {
    std::cout << i << ' ' << j << std::endl;
    i = i+1;
  }
  queue[r++] = i;
  queue[r++] = j;
  ga[i][j]=10;

  i = 190;
  j = image->w-10;
  while(raw[i][j] == 1)
    j = j-1;
  queue[r++] = i;
  queue[r++] = j;
  ga[i][j]=10;

  i = image->h-10;
  j = image->w-10;
  while(raw[i][j] == 1)
    j = j-1;
  queue[r++] = i;
  queue[r++] = j;
  ga[i][j]=10;

  while(l!=r)
  {
    int x=queue[l++];
    int y=queue[l++];
  //std::cout << l << ' ' << r << " OK1!\n";

    int xx = -1;
    int yy = 0;
    if((x+xx>=0) && (y+yy>=0) && (x+xx < image->h) && (y+yy < image->w)) {
      if((raw[x+xx][y+yy]==0) && (ga[x+xx][y+yy]==0)) {
        queue[r++] = x+xx;
        queue[r++] = y+yy;
        ga[x+xx][y+yy]=10;
      }
    }
  //std::cout << l << ' ' << r << " OK2!\n";

    xx = 1;
    yy = 0;
    if((x+xx>=0) && (y+yy>=0) && (x+xx < image->h) && (y+yy < image->w)) {
      if((raw[x+xx][y+yy]==0) && (ga[x+xx][y+yy]==0)) {
        queue[r++] = x+xx;
        queue[r++] = y+yy;
        ga[x+xx][y+yy]=10;
      }
    }
  //std::cout << l << ' ' << r << " OK3!\n";

    xx = 0;
    yy = -1;
    if((x+xx>=0) && (y+yy>=0) && (x+xx < image->h) && (y+yy < image->w)) {
      if((raw[x+xx][y+yy]==0) && (ga[x+xx][y+yy]==0)) {
        queue[r++] = x+xx;
        queue[r++] = y+yy;
        ga[x+xx][y+yy]=10;
      }
    }
  //std::cout << l << ' ' << r << " OK4!\n";

    xx = 0;
    yy = 1;
    if((x+xx>=0) && (y+yy>=0) && (x+xx < image->h) && (y+yy < image->w)) {
      if((raw[x+xx][y+yy]==0) && (ga[x+xx][y+yy]==0)) {
        queue[r++] = x+xx;
        queue[r++] = y+yy;
        ga[x+xx][y+yy]=10;
      }
    }
  //std::cout << l << ' ' << r << " OK5!\n";
  } // while(l!=r)

  for(int x=0; x< image->h; x++)
    for(int y=0; y<image->w; y++) {
      if(ga[x][y] == 10)
        ga[x][y] = 0;
      else
        ga[x][y] = 1;
    }
}

void SobelFluid_E::segmentImage(tifImage *image)
{
  std::cout << "rate: " << rate << std::endl;

  auto tf = new TaskFactory();
  auto gs = new AggressiveGS(20);
  auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
  auto tp0status = new __fluid__<bool>(tp0->isfinished());
  auto v0 = new ValveEQ<bool>(tp0status, true);
  tp0->run();

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
  ////<<<g1,{},{},call_num1>>>Gaussian_filter(image); // buyao

  ////<<<g1,{},{},{d1},{d2}>>>Gaussian_filter(image,call_num);
  auto tpb__g1 = std::bind(&SobelFluid_E::Gaussian_filter, this, image, std::placeholders::_1);
  auto tp__g1 = tf->newTask<decltype(tpb__g1), int*>("Gaussian_filter", {d1}, {d2}, tpb__g1, call_num1->p);
  Guard* g1 = gs->newGuard("Gaussian_filter", {}, {}, tp__g1, {});
  ////g1->set_root;
  g1->set_root();
  guard_log.push_back(g1);

  ////<<<g2,{v1(call_num1,image->size*rate)},{g1},call_num2>>>Sobel_filter(image); // buyao

  ////#pragma valve v0_ = v1.init(call_num1,image->size*rate);
  ////#pragma valve v1_ = v1.init(call_num1,image->size+1);
  auto v0_ = v1.init(call_num1,image->size*rate);
  auto v1_ = v1.init(call_num1,(image->h-10)*(image->w-10));
  ////<<<g2,{v0_},{v1_},{d2},{d3}>>Sobel_filter(image);
  auto tpb__g2 = std::bind(&SobelFluid_E::Sobel_filter, this, image);
  auto tp__g2 = tf->newTask<decltype(tpb__g2)>("Sobel_filter", {d2}, {d3}, tpb__g2);
  Guard* g2 = gs->newGuard("Sobel_filter", {v0_}, {v1_}, tp__g2, {});
  /////g2->set_leaf();
  g2->set_leaf();
  guard_log.push_back(g2);
  //Sobel_filter(image); 
  //gs->synctask(tp__g2);
  gs->sync(tp__g2);

}



////void SobelFluid::Gaussian_filter(tifImage *image)
void SobelFluid_E::Gaussian_filter(tifImage *image, int *call_num)
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

////void SobelFluid::Sobel_filter(tifImage *image)
void SobelFluid_E::Sobel_filter(tifImage *image)
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
  //printf("Sobel_filter end!!!\n");
  //printf("Sobel_filter end!!!\n");
}

//////////////////////////////////////////////////////////////Multi-threads/////////////////////////////////////

void SobelFluid_multi_E::segmentImage(tifImage *image)
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
    auto tpb__g1 = std::bind(&SobelFluid_multi_E::Gaussian_filter_threads, this, this, image, this->thread_num, th, std::placeholders::_1);
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
    auto tpb__g2 = std::bind(&SobelFluid_multi_E::Sobel_filter_threads, this, this, image, this->thread_num, th);
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
void SobelFluid_multi_E::Gaussian_filter_threads(SobelFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id, int *call_num)
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
void SobelFluid_multi_E::Gaussian_filter(tifImage *image, int num_thread, std::vector<int> *call_num)
{
  //std::cout << "Gaussian_filter start!!!\n";
/*
  double sigma_2 = sigma * sigma;
  double sum = 0.0;
  double *kernel = new double[121];

  for(int i=0;i<11;i++) {
    for(int j=0;j<11;j++) {
        kernel[11*i+j] = exp((-(i - 5) * (i - 5) - (j - 5) * (j - 5)) / (2 * sigma_2)) / (2 * 3.1415 * sigma_2);
        sum = sum + kernel[11*i+j];
    }
  }
  for(int i=0;i<121;i++)
    kernel[i] = kernel[i] / sum;
*/
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
  // std::cout << "Gaussian_filter end!!!\n";
  //printf("Gaussian_filter end %d!!!\n", *call_num);
}



////void SobelFluid::Sobel_filter(tifImage *image)
void SobelFluid_multi_E::Sobel_filter_threads(SobelFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id)
{
  std::cout << "Sobel_filter start!!!\n";
  /*
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
*/

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
  // std::cout << "Sobel_filter end!!!\n";
  //printf("Sobel_filter end!!!\n");
  //printf("Sobel_filter end!!!\n");
}

////void SobelFluid::Sobel_filter(tifImage *image)
void SobelFluid_multi_E::Sobel_filter(tifImage *image, int num_thread)
{
  //std::cout << "Sobel_filter start!!!\n";
  std::thread* threads = new std::thread[num_thread-1];
  for(int i=1; i<num_thread; i++) {
    //threads[i-1] = std::thread(Sobel_filter_threads, this, image, num_thread, i);
  }
  /*
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
*/

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
  //std::cout << "Sobel_filter end!!!\n";
  //printf("Sobel_filter end!!!\n");
  //printf("Sobel_filter end!!!\n");
}










void Gaussian_filter_threads22(SobelFluid_multi_E *sb, tifImage *image, double *kernel, int num_thread, int thread_id) {
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
    //(*call_num)+=image->w - 9;
  }
}


////void SobelFluid::Gaussian_filter(tifImage *image)
void SobelFluid_multi_E::Gaussian_filter_threads2(SobelFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id, int *call_num)
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

void Sobel_filter_threads22(SobelFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id) {
  double *kernel_x = sb->kernel_x;
  double *kernel_y = sb->kernel_y;
  int start_x = thread_id == 0 ? num_thread : thread_id;
  for(int x=start_x; x<image->h-1 ;x+=num_thread) {
    for(int y=1; y<image->w-1 ;y++) {
      sb->conv_sobel(image->gaussian, kernel_x, kernel_y, x, y, image->sobel);
    }
  }
}

////void SobelFluid::Sobel_filter(tifImage *image)
void SobelFluid_multi_E::Sobel_filter_threads2(SobelFluid_multi_E *sb, tifImage *image, int num_thread, int thread_id)
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
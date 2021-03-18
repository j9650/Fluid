#include "laplacian.h"
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
#include <math.h> 

using namespace example;	

//void print_to_logs(tifImage *graph, std::string st)
//{
  //graph->logs.push_back(st);
//}

int Laplacian::sobel(std::string in, std::string out) {

  image = new tifImage;
  std::cout << "pthread_self: "<< pthread_self() << std::endl;
  std::cout << "sobel num_thread: " << this->thread_num << "\n";

  std::cout << "OK!\n";
	InitImage(image,in);

  std::cout << "OK!\n";
  segmentImage(image);

  std::cout << "OK!\n";
  //testImage(image, out);
	//printf("Sobel!\n");
	return 0;
}

void Laplacian::InitImage(tifImage *image, std::string in)
{
	//std::ifstream file (in);
  	std::ifstream file;
  	file.open(in, std::ifstream::in);

  	file >> image->h;
  	file >> image->w;
    int dups = this->dups;
    std::cout << image->h * dups << ' ' << image->w << std::endl;
    image->size = image->w * image->h * dups;
    image->pixels = (double**)malloc(image->h * dups * sizeof(double*));
    image->gaussian = (double**)malloc(image->h * dups * sizeof(double*));
    image->sobel = (int**)malloc(image->h * dups * sizeof(int*));
    image->link = (int**)malloc(image->h * dups * sizeof(int*));
    image->segment = (int**)malloc(image->h * dups * sizeof(int*));
    for (int i = 0; i < image->h * dups; i++) {
      image->pixels[i] = (double*)malloc(image->w * sizeof(double));
      image->gaussian[i] = (double*)malloc(image->w * sizeof(double));
      image->sobel[i] = (int*)malloc(image->w * sizeof(int));
      image->link[i] = (int*)malloc(image->w * sizeof(int));
      image->segment[i] = (int*)malloc(image->w * sizeof(int));
    }



    for(int i=0; i<image->h; i++)
    {
      for(int j=0; j<image->w; j++)
      file >> image->pixels[i][j];
    }

    if (this->dups > 1) {
      for (int i = 0; i < image->h; i++){
        for (int j = 1; j < dups; j++){
          memcpy(image->pixels[j * image->h + i], image->pixels[i], sizeof(double) * image->w);
        }
      }
      image->h = image->h * dups;
    }

    double sigma_2 = sigma * sigma;
    double sum = 0.0;
    kernel = new double[121];

    for(int i=0;i<11;i++) {
      for(int j=0;j<11;j++) {
          kernel[11*i+j] = exp((-(i - 5) * (i - 5) - (j - 5) * (j - 5)) / (2 * sigma_2)) / (2 * 3.1415 * sigma_2);
          sum = sum + kernel[11*i+j];
      }
    }
    for(int i=0;i<121;i++)
      kernel[i] = kernel[i] / sum;

    kernel_x = new double[9];
    kernel_y = new double[9];
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

}

void Laplacian::segmentImage(tifImage *image)
{
  if (mode == 1 || mode == 2) {
    Gaussian_filter(image); 
  } else if (mode == 3 || mode == 4) {
    Mean_filter(image);
  }

  if (mode == 1 || mode == 3) {
    Sobel_filter(image);
  } else if (mode == 2 || mode == 4) {
    Laplacian_filter(image);
  }
  //for(int i=200; i<250; i++){
  //  for(int j=200; j<250; j++) {
  //    std::cout << image->sobel[i][j];
  //  }
  //  std::cout << std::endl;
  //}
  //Link_edge(image);
  //Segment(image);
  std::cout << "OK4!\n";

}

double Laplacian::conv(double **feature, double *kernel, int k_n, int x, int y)
{
  double sum = 0.0;
  for(int k=-k_n/2; k<k_n/2+1; k++)
    for(int m=-k_n/2; m<k_n/2+1; m++)
      sum = sum + kernel[(k+k_n/2)*k_n+m+k_n/2] * feature[x+k][y+m];
  return sum;
}

void Laplacian::conv_gaussian(double **feature, double *kernel, int x, int y, double **output)
{
  output[x][y] = conv(feature, kernel, 11, x, y);
}

void Laplacian::Gaussian_filter(tifImage *image)
{
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
      conv_gaussian(image->pixels, kernel, i, j, image->gaussian);
    }
  }
}


void Laplacian::conv_mean(double **feature, double *kernel, int k_size, int x, int y, double **output)
{
  output[x][y] = conv(feature, kernel, k_size, x, y);
}

void Laplacian::Mean_filter(tifImage *image)
{
  double sigma_2 = sigma * sigma;
  double sum = 0.0;
  int kernel_size = 11;
  double *kernel = new double[kernel_size * kernel_size];

  for(int i=0; i<image->h; i++) {
    for(int j=0; j<image->w; j++) {
      image->gaussian[i][j] = image->pixels[i][j];
    }
  }

  for(int i=0;i<kernel_size;i++) {
    for(int j=0;j<kernel_size;j++) {
        kernel[kernel_size*i+j] = 1.0 / (kernel_size * kernel_size);
    }
  }

  for(int i=(kernel_size / 2); i<image->h-(kernel_size / 2); i++) {
    for(int j=(kernel_size / 2); j<image->w-(kernel_size / 2); j++) {
      conv_mean(image->pixels, kernel, kernel_size, i, j, image->gaussian);
    }
  }
}

void Laplacian::conv_sobel(double **feature, double *kernel_x, double *kernel_y, int x, int y, int **output)
{
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
  double gx = conv(feature, kernel_x, 3, x, y);
  double gy = conv(feature, kernel_y, 3, x, y);
  double sum = sqrt(gx*gx+gy*gy);
  output[x][y] = int(round(sum * 256));
  if(output[x][y]>256) output[x][y] = 256;
  //if(sum>sobel_th)
  //  output[x][y] = 1.0;
  //else
  //  output[x][y] = 0.0;
}

void Laplacian::Sobel_filter(tifImage *image)
{
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
      conv_sobel(image->gaussian, kernel_x, kernel_y, x, y, image->sobel);
    }
  }
}

void Laplacian::conv_laplacian(double **feature, double *kernel, int x, int y, int **output)
{
  double sum = conv(feature, kernel, 3, x, y);
  output[x][y] = int(round(sum * 256));
  if(output[x][y]>256) output[x][y] = 256;
  //output[x][y] = conv(feature, kernel, 9, x, y);
}

void Laplacian::Laplacian_filter(tifImage *image)
{
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
      conv_laplacian(image->gaussian, kernel_x, x, y, image->sobel);
    }
  }
}

void Laplacian::Link_edge(tifImage *image)
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

void Laplacian::Segment(tifImage *image)
{
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

void Laplacian::testImage(tifImage *image, std::string out)
{
  //std::ifstream file (in);
  std::ifstream file;
  file.open(out, std::ifstream::in);

  int h,w,size;
  file >> h;
  file >> w;
  size = h * w;
  image->label = (int**)malloc(h*dups * sizeof(int*));
  for (int i = 0; i < h*dups; i++) {
    image->label[i] = (int*)malloc(w * sizeof(int));
  }

  for(int i=0; i<h; i++)
  {
    for(int j=0; j<w; j++)
      file >> image->label[i][j];
  }

  if (this->dups > 1) {
    for (int i = 0; i < h; i++){
      for (int j = 1; j < dups; j++){
        memcpy(image->label[j * h + i], image->label[i], sizeof(double) * image->w);
      }
    }
    h = h * dups;
  }

  int intersection=0;
  int uni = 0;
  double cc=0.0;
  for(int i=7; i<h-7; i++){
    for(int j=7; j<w-7; j++) {
      if(image->sobel[i][j] && image->label[i][j]) intersection++;
      if(image->sobel[i][j] || image->label[i][j]) uni++;

      int kkk = (image->sobel[i][j] - image->label[i][j]*256);
      cc += kkk*kkk;
    }
  }
  cc = cc / ((image->h-14)*(image->w-14));
  //for(int i=200; i<250; i++){
  //  for(int j=200; j<250; j++) {
  //    std::cout << image->label[i][j];
  //  }
  //  std::cout << std::endl;
  //}

  //double IOU = 1.0*double(intersection)/double(uni);
  //std::cout << "IOU: \t" << IOU << std::endl;
  //printf("IOU: \t%10f\n",IOU);
  //this->iou = IOU;
  std::cout << cc << " :cc" << std::endl;
  double PSNR = 10*log10(256*256/cc);
  std::cout << "PSNR: " << PSNR << std::endl;
  this->iou = PSNR;
}

//https://pdfs.semanticscholar.org/31b6/064341b46089fffa92425b291bfddc1af60e.pdf
//Use PSNR as the metric
/////////////////////////////////////////////////////////////////////////////////////////////////

////void SobelFluid::Gaussian_filter(tifImage *image)
void Gaussian_filter_threads1(Laplacian_multi *sb, tifImage *image, int num_thread, int thread_id)
{
  std::cout << "Gaussian_filter start!!!\n";

  for(int i=thread_id; i<image->h; i+=num_thread) {
    for(int j=0; j<image->w; j++) {
      image->gaussian[i][j] = image->pixels[i][j];
    }
  }
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
      {sb->conv_gaussian(image->pixels, kernel, i, j, image->gaussian);}
      //std::cout << "thread: " << thread_id << ", call_num: " << *call_num << std::endl;
    }
  }
}

////void SobelFluid::Sobel_filter(tifImage *image)
void Sobel_filter_threads1(Laplacian_multi *sb, tifImage *image, int num_thread, int thread_id)
{
  std::cout << "Sobel_filter start!!!\n";
  
  double *kernel_x = sb->kernel_x;
  double *kernel_y = sb->kernel_y;
  for(int x=thread_id; x<image->h-1 ;x+=num_thread) {
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

void Laplacian_multi::segmentImage(tifImage *image)
{
  std::cout << "segmentImage, num_thread: " << this->thread_num << "\n";
  Gaussian_filter(image, this->thread_num); 
  Sobel_filter(image, this->thread_num);
  //for(int i=200; i<250; i++){
  //  for(int j=200; j<250; j++) {
  //    std::cout << image->sobel[i][j];
  //  }
  //  std::cout << std::endl;
  //}
  //Link_edge(image);
  //Segment(image);
  std::cout << "OK4!\n";

}

////void SobelFluid::Gaussian_filter(tifImage *image)
void Laplacian_multi::Gaussian_filter(tifImage *image, int num_thread)
{
  std::cout << "Gaussian_filter start!!! num_thread: " << num_thread << "\n";

  std::thread* threads = new std::thread[num_thread-1];
  for(int i=1; i<num_thread; i++) {
    threads[i-1] = std::thread(Gaussian_filter_threads1, this, image, num_thread, i);
  }
  double *kernel = this->kernel;
  for(int i=0; i<image->h; i+=num_thread) {
    for(int j=0; j<image->w; j++) {
      image->gaussian[i][j] = image->pixels[i][j];
    }
  }

  int start_i = ((5 - 1) / num_thread) * num_thread;
  start_i = start_i >=5 ? start_i : start_i + num_thread;
  for(int i=start_i; i<image->h-5; i+=num_thread) {
    for(int j=5; j<image->w-5; j++) {
      ////#pragma call_num {conv_gaussian(image->pixels, kernel, i, j, image->gaussian);}
      {conv_gaussian(image->pixels, kernel, i, j, image->gaussian);}
      //std::cout << "thread: " << thread_id << ", call_num: " << *call_num << std::endl;
    }
  }
  for(int i=1; i<num_thread; i++) {
    threads[i-1].join();
  }
  //std::cout << "Gaussian_filter end!!!\n";
  //printf("Gaussian_filter end %d!!!\n", *call_num);
}

////void SobelFluid::Sobel_filter(tifImage *image)
void Laplacian_multi::Sobel_filter(tifImage *image, int num_thread)
{
  std::cout << "Sobel_filter start!!! num_thread: " << num_thread << "\n";
  std::thread* threads = new std::thread[num_thread-1];
  for(int i=1; i<num_thread; i++) {
    threads[i-1] = std::thread(Sobel_filter_threads1, this, image, num_thread, i);
  }

  double *kernel_x = this->kernel_x;
  double *kernel_y = this->kernel_y;
  for(int x=num_thread; x<image->h-1 ;x+=num_thread) {
    for(int y=1; y<image->w-1 ;y++) {
      ////#pragma call_num {conv_sobel(image->gaussian, kernel_x, kernel_y, x, y, image->sobel);}
      conv_sobel(image->gaussian, kernel_x, kernel_y, x, y, image->sobel);
      //(*call_num)++;}
    }
  }
  for(int i=1; i<num_thread; i++) {
    threads[i-1].join();
  }
  // std::cout << "Sobel_filter end!!!\n";
  //printf("Sobel_filter end!!!\n");
  //printf("Sobel_filter end!!!\n");
}
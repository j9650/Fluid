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

#include <sys/time.h>
//http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.302.9543&rep=rep1&type=pdf

#define MAX_COUNT 1200000
#define STABLE_TH 6
#define STABLE_DI 1
namespace example {


	class feature_map {
	public:
		int w;
		int h;
		int c;
		int n;
		int size;
		double *pixels;

		feature_map(int h_, int w_, int c_, int n_) : h(h_), w(w_), c(c_), n(n_) {
			size = h * w * c;
			printf("malloc %d %d %d %d\n", c,h,w,n);
			pixels = (double*)malloc(sizeof(double)*c*h*w*n);
			memset(pixels, 0, sizeof(double)*c*h*w*n);
		}
	};

	class weights {
	public:
		int w;
		int h;
		int ic;
		int oc;
		int size;
		double *pixels;
		double *bias;

		weights(int h_, int w_, int ic_, int oc_) : h(h_), w(w_), ic(ic_), oc(oc_) {
			size = h * w * ic * oc;
			pixels = (double*)malloc(sizeof(double)*size);
			bias = (double*)malloc(sizeof(double)*oc);
		}
	};

	class conv_layer {
	public:
		int ih;
		int oh;
		int iw;
		int ow;
		int ic;
		int oc;
		int kw;
		int kh;
		bool valid;
		int stride;
		int pad;

		feature_map *feature_in;
		feature_map *feature_out;
		weights *weight;
		double *column;

		conv_layer(int h_, int w_, int ic_, int oc_, int kw_, int kh_, int s_, bool valid_, feature_map *input, feature_map *output) :
			ih(h_), iw(w_), ic(ic_), oc(oc_), kw(kw_), kh(kh_), stride(s_), feature_in(input), feature_out(output), valid(valid_)
		{
			assert(ic_ == input->c);
			assert(oc_ == output->c);

			weight = new weights(kh, kw, ic, oc);
			if (valid) {
				//oh = ih - kh + 1;
				//ow = iw - kw + 1;
				oh = (ih - kh) / stride + 1;
				ow = (iw - kw) / stride + 1;
				pad = 0;
			} else {
				//oh = ih;
				//ow = iw;
				oh = (ih - 1) / stride + 1;
				ow = (iw - 1) / stride + 1;
				pad = ((ih - 1) / stride) * stride + kh - ih;
			}

			column = (double*)malloc(sizeof(double)*feature_out->h*feature_out->w*kh*kw*ic);
		}

		virtual void convolution();
  		virtual void im2col();
  		virtual void matmul();
  		virtual void addbias();
	};


	class CNN {
	public:
		//static std::string loopindicator;
		double rate;
		double sum;
		int dups;
		double sigma;
		double iou;
		int batch_size;
		int h,w,c;
		bool relu;
		bool pad;

		std::string weight_file;
		int num_images;
		int input_height;
		int input_width;
		int input_channel;
		std::vector<int> labels;
		std::vector<int> predicts;

		double accuracy;

		// feature maps:
		feature_map* input;  // 32 32 3
		feature_map* conv2d_1;  // 32 32 32
		feature_map* activation_1;  // 32 32 32
		feature_map* conv2d_2;  // 30 30 32
		feature_map* activation_2;  // 30 30 32
		feature_map* max_pooling2d_1;  // 15 15 32
		feature_map* conv2d_3;  // 15 15 64
		feature_map* activation_3;  // 15 15 64
		feature_map* conv2d_4;  // 13 13 64
		feature_map* activation_4;  // 13 13 64
		feature_map* max_pooling2d_2;  // 6 6 64
		feature_map* flatten_;  // 2304
		feature_map* dense_1;  // 512
		feature_map* activation_5;  // 512
		feature_map* dense_2;  // 10
		feature_map* activation_6;  // 10

		// dense_layer weights:
		weights* dense_weights1;
		weights* dense_weights2;
		
		std::vector<weights*> all_weights_list;
		std::vector<std::string> all_weights_name;

		void alexnet_flower(int h_, int w_);
		void mnist(int h_, int w_, int c_, int batch_size_);

		virtual void cnn(std::string input_path, std::string in); // done
		void load_weights(std::string weight_filename); // done
		void load_weights_mnist(std::string weight_filename); // done
		void load_image(std::string image_filename); // done
		void load_image_mnist(std::string image_filename, double* pixels); // done

		void load_image_lenet(std::string image_filename, double* pixels); // done
		void load_image_imagenet(std::string image_filename, double* pixels); // done
		//virtual void load_weights_lenet(std::string weight_filename); // done
		//int inference(); // done
		virtual std::vector<int> inference(); // done
		void test_result(); // done

  		//virtual void conv_layer();
  		void relu_layer(feature_map* input, feature_map* output); // done
  		void softmax_layer(feature_map* input, feature_map* output); // done
  		void maxpool_layer(feature_map* input, feature_map* output, int k, int stride); // done
  		void dense_layer(feature_map* input, feature_map* output, weights* dense_weights); // done
  		void dense_layer(feature_map* input, feature_map* output, weights* dense_weights, int *iiiii); // done
  		void flatten(feature_map* input, feature_map* output);
  		void convolution_layer(feature_map* input, feature_map* output, weights* conv_weights);
  		void crop_layer(feature_map* input, feature_map* output, int cut);
  		void concat_layer(feature_map* input1, feature_map* input2, feature_map* output);

  		// functions for Fluid:
  		void relu_layer_E(feature_map* input, feature_map* output, int* iiiii);
  		void softmax_layer_E(feature_map* input, feature_map* output, int* iiiii);
  		void maxpool_layer_E(feature_map* input, feature_map* output, int k, int stride, int* iiiii);
  		void dense_layer_E(feature_map* input, feature_map* output, weights* dense_weights, double scale, int *iiiii);
  		void flatten_E(feature_map* input, feature_map* output, int* iiiii);
  		void convolution_layer_E(feature_map* input, feature_map* output, weights* conv_weights, double scale, int* iiiii);
  		void crop_layer_E(feature_map* input, feature_map* output, int cut, int* iiiii);
  		void concat_layer_E(feature_map* input1, feature_map* input2, feature_map* output, int* iiiii);


		CNN(int h_, int w_, int c_, int batch_size_)
		{
			batch_size = batch_size_;  
		}

	//private:
		// conv_layers:
		conv_layer *conv2d_layer1;
		conv_layer *conv2d_layer2;
		conv_layer *conv2d_layer3;
		conv_layer *conv2d_layer4;

	};


	class CNN_lenet : public CNN {
	public:
		// feature maps:
		feature_map* input;  // 28 28 1
		feature_map* conv2d_1;  // 26 26 512 #weight1
		feature_map* activation_1;  // 26 26 512
		feature_map* max_pooling2d_1;  // 13 13 512
		feature_map* conv2d_22;  // 11 11 512 #weight2
		feature_map* activation_2;  // 11 11 512
		feature_map* max_pooling2d_2;  // 5 5 512

		feature_map* flatten_;  // 12800
		feature_map* dense_1;  // 120 #weight3
		feature_map* activation_3;  // 120
		feature_map* dense_2;  // 84 #weight4
		feature_map* activation_4;  // 84
		feature_map* dense_3;  // 10 # weight5
		feature_map* activation_5;  // 10

		// conv_layer weights:
		weights* conv_weights1;
		weights* conv_weights2;
		// dense_layer weights:
		weights* dense_weights1;
		weights* dense_weights2;
		weights* dense_weights3;

		void lenet(int h_, int w_, int c_, int batch_size_);

		virtual void cnn(std::string input_path, std::string in);
		void load_weights_lenet(std::string weight_filename);
		virtual std::vector<int> inference();


		CNN_lenet(int h_, int w_, int c_, int batch_size_) : CNN(h_, w_, c_, batch_size_)
		{
			batch_size = batch_size_;
			input_height = h_;
			input_width = w_;
			input_channel = c_; 
			relu = false;
			pad = false;
		}

		// conv_layers:
		conv_layer *conv2d_layer1;
		conv_layer *conv2d_layer2;
		conv_layer *conv2d_layer3;
		conv_layer *conv2d_layer4;

	};

	class CNN_vggnet : public CNN {
	public:
		// feature maps:
		feature_map* input;  // 28 28 1
		feature_map* conv2d_0;  // 26 26 512 #weight1
		feature_map* conv2d_1;  // 26 26 512 #weight1
		feature_map* max_pooling2d_0;  // 13 13 512
		feature_map* conv2d_2;  // 26 26 512 #weight1
		feature_map* conv2d_3;  // 26 26 512 #weight1
		feature_map* max_pooling2d_1;  // 13 13 512
		feature_map* conv2d_4;  // 26 26 512 #weight1
		feature_map* conv2d_5;  // 26 26 512 #weight1
		feature_map* conv2d_6;  // 26 26 512 #weight1
		feature_map* max_pooling2d_2;  // 13 13 512
		feature_map* conv2d_7;  // 26 26 512 #weight1
		feature_map* conv2d_8;  // 26 26 512 #weight1
		feature_map* conv2d_9;  // 26 26 512 #weight1
		feature_map* max_pooling2d_3;  // 13 13 512
		feature_map* conv2d_10;  // 26 26 512 #weight1
		feature_map* conv2d_11;  // 26 26 512 #weight1
		feature_map* conv2d_12;  // 26 26 512 #weight1
		feature_map* max_pooling2d_4;  // 13 13 512

		feature_map* flatten_;  // 12800
		feature_map* dense_1;  // 120 #weight3
		feature_map* activation_1;  // 10
		feature_map* dense_2;  // 84 #weight4
		feature_map* activation_2;  // 10
		feature_map* dense_3;  // 10 # weight5
		feature_map* activation_3;  // 10

		// conv_layer weights:
		weights* conv_weights0;
		weights* conv_weights1;
		weights* conv_weights2;
		weights* conv_weights3;
		weights* conv_weights4;
		weights* conv_weights5;
		weights* conv_weights6;
		weights* conv_weights7;
		weights* conv_weights8;
		weights* conv_weights9;
		weights* conv_weights10;
		weights* conv_weights11;
		weights* conv_weights12;
		// dense_layer weights:
		weights* dense_weights1;
		weights* dense_weights2;
		weights* dense_weights3;


		void vggnet(int h_, int w_, int c_, int batch_size_);

		virtual void cnn(std::string input_path, std::string in); //done
		void load_weights_vggnet(std::string weight_filename);
		virtual std::vector<int> inference();


		CNN_vggnet(int h_, int w_, int c_, int batch_size_) : CNN(h_, w_, c_, batch_size_)
		{
			batch_size = batch_size_;
			input_height = h_;
			input_width = w_;
			input_channel = c_; 
			relu = true;
			pad = true;
		}

		// conv_layers:
		conv_layer *conv2d_layer1;
		conv_layer *conv2d_layer2;
		conv_layer *conv2d_layer3;
		conv_layer *conv2d_layer4;

	};

	class CNN_squeezenet : public CNN {
	public:
		// feature maps:
		feature_map* input;  // 28 28 1
		feature_map* conv2d_1_e1;  // 28 28 256 #weight1
		feature_map* crop_1;  // 26 26 256
		feature_map* conv2d_1_e3;  // 26 26 256 #weight2
		feature_map* concat_1;  // 26 26 512
		feature_map* activation_1;  // 26 26 512
		feature_map* max_pooling2d_1;  // 13 13 512


		feature_map* conv2d_2_s1;  // 13 13 64 #weight3
		feature_map* activation_2;  // 13 13 64
		feature_map* conv2d_2_e1;  // 13 13 256 #weight4
		feature_map* crop_2;  // 11 11 256
		feature_map* conv2d_2_e3;  // 11 11 256 #weight5
		feature_map* concat_2;  // 11 11 512
		feature_map* activation_3;  // 11 11 512
		feature_map* max_pooling2d_2;  // 5 5 512

		feature_map* flatten_;  // 12800
		feature_map* dense_1;  // 120 #weight3
		feature_map* activation_4;  // 120
		feature_map* dense_2;  // 84 #weight4
		feature_map* activation_5;  // 84
		feature_map* dense_3;  // 10 # weight5
		feature_map* activation_6;  // 10

		// conv_layer weights:
		weights* conv_weights1;
		weights* conv_weights2;
		weights* conv_weights3;
		weights* conv_weights4;
		weights* conv_weights5;
		// dense_layer weights:
		weights* dense_weights1;
		weights* dense_weights2;
		weights* dense_weights3;

		void squeezenet(int h_, int w_, int c_, int batch_size_);

		virtual void cnn(std::string input_path, std::string in);
		void load_weights_squeezenet(std::string weight_filename);
		virtual std::vector<int> inference();


		CNN_squeezenet(int h_, int w_, int c_, int batch_size_) : CNN(h_, w_, c_, batch_size_)
		{
			batch_size = batch_size_;
			input_height = h_;
			input_width = w_;
			input_channel = c_;    
			relu = false;
			pad = false;
		}

		// conv_layers:
		conv_layer *conv2d_layer1;
		conv_layer *conv2d_layer2;
		conv_layer *conv2d_layer3;
		conv_layer *conv2d_layer4;

	};
}
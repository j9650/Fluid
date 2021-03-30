#pragma once
#include "../fluid/guard.h"
#include "../fluid/valve.h"
#include "../fluid/fluid.h"
#include "../fluid/guardscheduler.h"
#include "cnn.h"
namespace example {


	class conv_layerFluid_E {
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

		conv_layerFluid_E(int h_, int w_, int ic_, int oc_, int kw_, int kh_, int s_, bool valid_, feature_map *input, feature_map *output) :
			//conv_layer(h_, w_, ic_, oc_, kw_, kh_, s_, valid_, input, output)
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

		virtual void convolution(int iiiii);
  		virtual void im2col(int * call_num);
  		virtual void matmul(int * call_num);
  		virtual void addbias(int * call_num);

  		//fluid entity
		ValveGT<int> v1;
		double rate;
	};

	class CNNFluid_E {
	public:
		std::vector<Guard*> guard_log;
		double sum;
		int dups;
		double sigma;
		double iou;
		int batch_size;

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
		
		void alexnet_flower_E(int h_, int w_, double rate_);
		void mnist_E(int h_, int w_, int c_, int batch_size_, double rate_);

		void cnn(std::string input_path, std::string in); // done
		void load_weights(std::string weight_filename); // done
		void load_weights_mnist(std::string weight_filename); // done
		void load_image(std::string image_filename); // done
		void load_image_mnist(std::string image_filename, double* pixels); // done
		//int inference(); // done
		std::vector<int> inference();
		void test_result(); // done
  		//virtual void conv_layer();
  		virtual void relu_layer(feature_map* input, feature_map* output); // done
  		virtual void relu_layer_E(feature_map* input, feature_map* output, int *iiiii); // done
  		virtual void softmax_layer(feature_map* input, feature_map* output); // done
  		virtual void softmax_layer_E(feature_map* input, feature_map* output, int *iiiii); // done
  		virtual void maxpool_layer(feature_map* input, feature_map* output, int k, int stride); // done
  		virtual void dense_layer(feature_map* input, feature_map* output, weights* dense_weights); // done
  		virtual void dense_layer_E(feature_map* input, feature_map* output, weights* dense_weights, double scale, int *iiiii); // done
  		virtual void flatten(feature_map* input, feature_map* output);
  		
		CNNFluid_E(int h_, int w_, int c_, int batch_size_, double rate_) // : CNN(h_, w_) 
		{
			batch_size = batch_size_;
			mnist_E(h_, w_, c_, batch_size_, rate);

		}

  		//fluid entity
		ValveGT<int> v1;
		double rate;

		// conv_layers:
		conv_layer *conv2d_layer1;
		conv_layerFluid_E *conv2d_layer2;
		conv_layer *conv2d_layer3;
		conv_layer *conv2d_layer4;

	};

	class CNN_lenetFluid_E : public CNN_lenet {
	public:
		std::vector<Guard*> guard_log;

		virtual std::vector<int> inference();
  		
		CNN_lenetFluid_E(int h_, int w_, int c_, int batch_size_, double rate_) : CNN_lenet(h_, w_, c_, batch_size_)
		{
			rate = rate_;
		}

  		//fluid entity
		ValveGT<int> v1;
		double rate;

	};

	class CNN_vggnetFluid_E : public CNN_vggnet {
	public:
		std::vector<Guard*> guard_log;

		virtual std::vector<int> inference();
  		
		CNN_vggnetFluid_E(int h_, int w_, int c_, int batch_size_, double rate_) : CNN_vggnet(h_, w_, c_, batch_size_)
		{
			rate = rate_;
		}

  		//fluid entity
		ValveGT<int> v1;
		double rate;

	};

	class CNN_squeezenetFluid_E : public CNN_squeezenet {
	public:
		std::vector<Guard*> guard_log;

		virtual std::vector<int> inference();
  		
		CNN_squeezenetFluid_E(int h_, int w_, int c_, int batch_size_, double rate_) : CNN_squeezenet(h_, w_, c_, batch_size_)
		{
			rate = rate_;
		}

  		//fluid entity
		ValveGT<int> v1;
		double rate;

	};
};
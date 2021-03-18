#include "cnn.h"
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
#include <string.h>

using namespace example;	

void print_feature(feature_map *map, std::string layer_name) {
  int n = map->n;
  double dist = 0.0;
  for (int c = 0; c < map->c; c++) {
    for (int h = 0; h < map->h; h++) {
      for (int w = 0; w < map->w; w++) {
        dist += abs(map->pixels[(((n-1) * map->c + c) * map->h + h) * map->w + w] - map->pixels[(((n-2) * map->c + c) * map->h + h) * map->w + w]);
      }
    }
  }
  FILE *f=fopen("wwcao.txt","a");
  int index = map->n * map->c * map->h * map->w - 5;
  fprintf(f, "%s %p %d %d %d %d %f %f\n", layer_name.c_str(), map->pixels, map->n, map->c, map->h, map->w, dist, map->pixels[index]);
  fclose(f);
}
//void print_to_logs(tifImage *graph, std::string st)
//{
  //graph->logs.push_back(st);
//}

void CNN::cnn(std::string input_path, std::string in) {
  std::cout << "CNN::cnn\n";
  mnist(h, w, c, batch_size);

  std::ifstream input_file;
  input_file.open(input_path + in, std::ifstream::in);

  std::string weight_filename;
  input_file >> weight_filename;
  ////load_weights(input_path + weight_filename);
  load_weights_mnist(input_path + weight_filename);
  std::cout << "succ loaded weights\n";

  input_file >> num_images;
  //file >> image->w;
  input_file >> input_height;
  input_file >> input_width;
  input_file >> input_channel;

  std::string image_filename;
  int label;
  std::vector<int> predict;
  //predicts.resize()

  // inference the images one by one
  double* pixels;
  for (int i = 0; i * batch_size < num_images; i++) {
    pixels = input->pixels;
    for (int j = 0; j + i < num_images && j < batch_size; j++) {
      input_file >> image_filename;
      input_file >> label;
      labels.push_back(label);
  
      load_image_mnist(input_path + image_filename, pixels);
  
      std::cout << "loaded image: " << input_path + image_filename << ' ' << label << std::endl;
      pixels += input->size;
    }

    predict = inference();
    //std::cout << "predicted: " << predict << std::endl;
    //predicts.push_back(predict);
    predicts.insert(predicts.end(), predict.begin(), predict.end());
    //if (i>10) break;
    break;
  }

  //test_result();
}

void CNN::test_result() {
  double correct = 0.0;
  double total = 0.0;
  double accuracy = 0.0;

  for (int i = 0; i < labels.size(); i++) {
    total = total + 1.0;
    if (labels[i] == predicts[i]) correct = correct + 1;
    //std::cout << labels[i] << ' ' << predicts[i] << std::endl;
  }

  accuracy = correct / total;
  std::cout << "accuracy: " << accuracy << std::endl;
  this->accuracy = accuracy;
}

void CNN::load_image(std::string image_filename) {
  // load image
  std::ifstream image_file;
  image_file.open(image_filename, std::ifstream::in);
  for (int c = 0; c < input_channel; c++) {
    for (int h = 0; h < input_height; h++) {
      for (int w = 0; w < input_width; w++) {
        image_file >> input->pixels[((c * input_height) + h) * input_width + w];
      }
    }
  }
}

void CNN::load_image_mnist(std::string image_filename, double* pixels) {
  // load image
  std::ifstream image_file;
  image_file.open(image_filename, std::ifstream::in);
  for (int c = 0; c < input_channel; c++) {
    for (int h = 0; h < input_height; h++) {
      for (int w = 0; w < input_width; w++) {
        image_file >> pixels[((c * input_height) + h) * input_width + w];
      }
    }
  }
}

void CNN::load_image_lenet(std::string image_filename, double* pixels) {
  // load image
  std::cout << "CNN::load_image_lenet " << image_filename << std::endl;
  std::ifstream image_file;
  image_file.open(image_filename, std::ifstream::in);
  //std::cout << "33n: " << 0 << " oc: " << input_channel << " space: " << input_height*input_width << std::endl;
  for (int c = 0; c < input_channel; c++) {
    for (int h = 0; h < input_height; h++) {
      for (int w = 0; w < input_width; w++) {
        image_file >> pixels[((c * input_height) + h) * input_width + w];
        //std::cout << pixels[((c * input_height) + h) * input_width + w]; // << std::endl;
      }
    }
  }
  //std::cout << "CNN::load_image_lenet " << *pixels << std::endl;
}

void CNN::load_weights_mnist(std::string weight_filename) {
  std::cout << "CNN::load_weights\n";
  std::ifstream weight_file;
  weight_file.open(weight_filename, std::ifstream::in);

  int ic = 784;
  int oc = 1024;
  int kw = 3;
  int kh = 3;
  std::cout << "CNN::load_weights init " << weight_filename << "\n";

  //conv2d_layer1
  double *weights;
  double *bias;


  //dense_layer1
  ic = dense_weights1->ic; oc = dense_weights1->oc; kw = 1; kh = 1;
  weights = dense_weights1->pixels;
  bias = dense_weights1->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }

  //dense_layer2
  ic = dense_weights2->ic; oc = dense_weights2->oc; kw = 1; kh = 1;
  weights = dense_weights2->pixels;
  bias = dense_weights2->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }

}
void CNN::load_weights(std::string weight_filename) {
  std::ifstream weight_file;
  weight_file.open(weight_filename, std::ifstream::in);

  int ic = 3;
  int oc = 32;
  int kw = 3;
  int kh = 3;

  //conv2d_layer1
  double *weights = conv2d_layer1->weight->pixels;
  double *bias = conv2d_layer1->weight->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }

  //conv2d_layer2
  ic = 32; oc = 32; kw = 3; kh = 3;
  weights = conv2d_layer2->weight->pixels;
  bias = conv2d_layer2->weight->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }

  //conv2d_layer3
  ic = 32; oc = 64; kw = 3; kh = 3;
  weights = conv2d_layer3->weight->pixels;
  bias = conv2d_layer3->weight->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }
  //conv2d_layer4
  ic = 64; oc = 64; kw = 3; kh = 3;
  weights = conv2d_layer4->weight->pixels;
  bias = conv2d_layer4->weight->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }
  //dense_layer1
  ic = dense_weights1->ic; oc = 512; kw = 1; kh = 1;
  weights = dense_weights1->pixels;
  bias = dense_weights1->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }
  //dense_layer2
  ic = dense_weights2->ic; oc = 10; kw = 1; kh = 1;
  weights = dense_weights2->pixels;
  bias = dense_weights2->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }

}
/*
int CNN::inference() {

  // structure of CNN:
  conv2d_layer1->convolution();
  relu_layer(conv2d_1, activation_1);
  conv2d_layer2->convolution();
  relu_layer(conv2d_2, activation_2);
  maxpool_layer(activation_2, max_pooling2d_1, 2, 2);
  conv2d_layer3->convolution();
  relu_layer(conv2d_3, activation_3);
  conv2d_layer4->convolution();
  relu_layer(conv2d_4, activation_4);
  maxpool_layer(activation_4, max_pooling2d_2, 2, 2);
  flatten(max_pooling2d_2, flatten_);
  //for (int i = 0; i < flatten_->size; i++) {
  //  std::cout << flatten_->pixels[i] << std::endl;
  //}
  dense_layer(flatten_, dense_1, dense_weights1);
  //for (int i = 0; i < dense_1->size; i++) {
  //  std::cout << dense_1->pixels[i] << std::endl;
  //}
  //std::cout << "====================================================" << std::endl;
  relu_layer(dense_1, activation_5);
  //for (int i = 0; i < activation_5->size; i++) {
  //  std::cout << activation_5->pixels[i] << std::endl;
  //}
  //std::cout << "====================================================" << std::endl;
  dense_layer(activation_5, dense_2, dense_weights2);
  //for (int i = 0; i < dense_2->size; i++) {
  //  std::cout << dense_2->pixels[i] << std::endl;
  //}
  //std::cout << "====================================================" << std::endl;


  softmax_layer(dense_2, activation_6);

  //for (int i = 0; i < activation_6->size; i++) {
  //  std::cout << activation_6->pixels[i] << std::endl;
  //}

  double max = 0.0;
  int catogary;
  for(int i = 0; i < activation_6->size; i++) {
    if (activation_6->pixels[i] > max) {
      max = activation_6->pixels[i];
      catogary = i;
    }
  }

  return catogary;
}*/

std::vector<int> CNN::inference() {

  int *iiiii=new int(0);
  //flatten(input, flatten_);
  dense_layer(input, dense_1, dense_weights1, iiiii);
  //for (int i = 0; i < dense_1->size; i++) {
  //  std::cout << dense_1->pixels[i] << std::endl;
  //}
  //std::cout << "====================================================" << std::endl;
  relu_layer(dense_1, activation_5);
  //for (int i = 0; i < activation_5->size; i++) {
  //  std::cout << activation_5->pixels[i] << std::endl;
  //}
  iiiii=new int(0);
  //std::cout << "====================================================" << std::endl;
  dense_layer(activation_5, dense_2, dense_weights2, iiiii);
  //for (int i = 0; i < dense_2->size; i++) {
  //  std::cout << dense_2->pixels[i] << std::endl;
  //}
  //std::cout << "====================================================" << std::endl;


  softmax_layer(dense_2, activation_6);

  //for (int i = 0; i < activation_6->size; i++) {
  //  std::cout << activation_6->pixels[i] << std::endl;
  //}

  double max;
  int catogary_res;
  std::vector<int> catogary;
  double* activation_6_pixels = activation_6->pixels;
  for (int n = 0; n < activation_6->n; n++) {
    max = 0.0;
    for(int i = 0; i < activation_6->size; i++) {
      if (*activation_6_pixels > max) {
        max = *activation_6_pixels;
        catogary_res = i;
      }
      activation_6_pixels++;
    }
    //printf("%d\n", catogary_res);
    catogary.push_back(catogary_res);
  }

  return catogary;
}

void CNN::softmax_layer(feature_map* input, feature_map* output) {
  assert(input->size == output->size);

  for (int n = 0; n < input->n; n++) {
    double total = 0.0;
    for (int i = 0; i < input->size; i++) {
      total += exp(input->pixels[n * input->size + i]);
    }
    for (int i = 0; i < output->size; i++) {
      output->pixels[n * output->size + i] = exp(input->pixels[n * input->size + i]) / total;
    }
  }
  //print_feature(output, "softmax_layer output");
}

void CNN::relu_layer(feature_map* input, feature_map* output) {
  std::cout << input->size << ' ' << output->size << std::endl;
  assert(input->size == output->size);

  for (int i = 0; i < output->size * output->n; i++) {
    output->pixels[i] = input->pixels[i] > 0 ? input->pixels[i] : 0;
  }
  //print_feature(output, "relu_layer output");
}

void CNN::maxpool_layer(feature_map* input, feature_map* output, int k, int stride) {
  assert(output->h == (input->h - k) / stride + 1);
  assert(output->c == input->c);

  double max;
  double *ans = output->pixels;
  for (int n = 0; n < output->n; n++) {
    //ans = output->pixels
    for (int c = 0; c < output->c; c++) {
      for (int h = 0; h < output->h; h++) {
        for (int w = 0; w < output->w; w++) {
          max = - 1000000.0;
          for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
              double tmp = input->pixels[((n * input->c + c) * input->h + h * stride + i) * input->w + w * stride + j];
              max = max > tmp ? max : tmp;
            }
          }
          *ans = max;
          ans++;
        }
      }
    }
  }
  //print_feature(output, "maxpool_layer output");
}

void CNN::dense_layer(feature_map* input, feature_map* output, weights* dense_weights) {
  assert(input->c == dense_weights->ic);
  assert(output->c == dense_weights->oc);
  assert(input->h == 1);
  assert(output->h == 1);
  assert(dense_weights->h == 1);
  assert(input->w == 1);
  assert(output->w == 1);
  assert(dense_weights->w == 1);

  for (int oc = 0; oc < output->c; oc++) {
    output->pixels[oc] = dense_weights->bias[oc];
  }
  //std::cout << "dense weights: " << std::endl;
  double *weights = dense_weights->pixels;
  for (int oc = 0; oc < output->c; oc++) {
    for (int ic = 0; ic < input->c; ic++) {
      output->pixels[oc] += input->pixels[ic] * (*weights);
      //std::cout << *weights << std::endl;
      weights++; 
    }
  }
  //std::cout << "====================================================" << std::endl;
}

void CNN::dense_layer(feature_map* input, feature_map* output, weights* dense_weights, int *iiiii) {
  //struct timeval t0, t1;
  //gettimeofday(&t0, 0);

  assert(input->c == dense_weights->ic);
  assert(output->c == dense_weights->oc);
  assert(input->h == 1);
  assert(output->h == 1);
  assert(dense_weights->h == 1);
  assert(input->w == 1);
  assert(output->w == 1);
  assert(dense_weights->w == 1);
  assert(input->n == output->n);

  double *output_feature = output->pixels;
  for (int i = 0; i < output->n; i++) {
    for (int oc = 0; oc < output->c; oc++) {
      //output->pixels[oc] = dense_weights->bias[oc];
      output_feature[oc] = dense_weights->bias[oc];
    }
    output_feature += output->c;
  }
  //std::cout << "dense weights: " << std::endl;
  //gettimeofday(&t1, 0);
  //long elapsed = (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;
  //std::cout << "CNN::dense_layer end " << elapsed << std::endl;

  //double *weights = dense_weights->pixels;
  //double *input_feature = input->pixels;
  //for (int oc = 0; oc < output->c; oc++) {
  //  for (int ic = 0; ic < input->c; ic++) {
  //    input_feature = input->pixels + ic;
  //    output_feature = output->pixels + oc;
  //    for (int i = 0; i < output->n; i++) {
  //      *output_feature += *input_feature * (*weights);
  //      input_feature += input->c;
  //      output_feature += output->c;
  //      //std::cout << *weights << std::endl;
  //    }
  //    weights++;
  //  }
  //}

  double *weights = dense_weights->pixels;
  double *input_feature = input->pixels;
  for (*iiiii = 0; *iiiii < output->c; (*iiiii)++) {
    //std::cout << *iiiii << std::endl;
    for (int ic = 0; ic < input->c; ic++) {
      input_feature = input->pixels + ic;
      output_feature = output->pixels + (*iiiii);
      for (int i = 0; i < output->n; i++) {
        //std::cout << output_feature << ' ' << input_feature << ' ' << i << std::endl;
        *output_feature += *input_feature * (*weights);
        input_feature += input->c;
        output_feature += output->c;
        //std::cout << *weights << std::endl;
      }
      
      weights++;
    }
  }
  //print_feature(output, "dense_layer output");
  //std::cout << "====================================================" << std::endl;
}

void CNN::flatten(feature_map* input, feature_map* output) {
  assert(input->size == output->size);

  double *flat = output->pixels;
  
  for (int n = 0; n < input->n; n++) {
    for (int h = 0; h < input->h; h++) {
      for (int w = 0; w < input->w; w++) {
        for (int c = 0; c < input->c; c++) {
        *flat = input->pixels[((n * input->c + c) * input->h + h) * input->w + w];
        flat++;
        }
      }
    }
  }
  //for (int p = 0; p < input->size; p++) {
  //  output->pixels[p] = input->pixels[p];
  //}
  //print_feature(output, "flatten output");
}

void CNN::crop_layer(feature_map* input, feature_map* output, int cut) {
  assert(input->h = output->h + 2*cut);
  assert(input->w = output->w + 2*cut);
  
  double *output_feature = output->pixels;
  for (int n = 0; n < output->n; n++) {
    for (int c = 0; c < output->c; c++) {
      double *input_feature = input->pixels + ((n * input->c + c) * input->h + cut) * input->w + cut;
      for (int h = 0; h < output->h; h++) {
        memcpy(output_feature, input_feature, sizeof(double) * output->w);
        input_feature += input->w;
        output_feature += output->w;
      }
      assert(input_feature == input->pixels + ((n * input->c + c) * input->h + input->h - cut) * input->w + cut);
    }
  }
  //print_feature(output, "crop_layer output");
}

void CNN::concat_layer(feature_map* input1, feature_map* input2, feature_map* output) {
  assert(input1->n = input2->n);
  assert(input2->n = output->n);
  assert(input1->h = input2->h);
  assert(input2->h = output->h);
  assert(input1->w = input2->w);
  assert(input2->w = output->w);
  assert(output->size == input1->size + input2->size);
  std::cout << "CNN::concat_layer" << input1->n << ' ' << input2->n << ' ' << output->n << std::endl;

  double *output_feature = output->pixels;
  double *input1_feature = input1->pixels;
  double *input2_feature = input2->pixels;
  for (int n = 0; n < input1->n; n++) {
    memcpy(output_feature, input1_feature, sizeof(double) * input1->size);
    //printf("%lf %lf\n", output_feature[100], input1_feature[100]);
    output_feature += input1->size;
    input1_feature += input1->size;
    memcpy(output_feature, input2_feature, sizeof(double) * input2->size);
    //printf("%lf %lf\n", output_feature[110], input2_feature[110]);
    output_feature += input2->size;
    input2_feature += input2->size;
  //print_feature(input1, "Concat input1");
  //print_feature(input2, "Concat input2");
  //print_feature(output, "Concat output");
  }
  assert(output_feature == output->pixels + output->n * output->size);
  assert(input1_feature == input1->pixels + input1->n * input1->size);
  assert(input2_feature == input2->pixels + input2->n * input2->size);
  //print_feature(output, "concat_layer output");
}

void CNN::convolution_layer(feature_map* input, feature_map* output, weights* conv_weights) {
  int n = input->n;
  int ic = input->c;
  int ih = input->h;
  int iw = input->w;
  int oc = output->c;
  int oh = output->h;
  int ow = output->w;

  std::cout << "CNN::convolution_layer1" << std::endl;
  double *output_feature = output->pixels;
  int space = output->h * output->w;
  //std::cout << "n: " << output->n << " oc: " << output->c << " space: " << space << std::endl;
  for (int i = 0; i < output->n; i++) {
    //std::cout << "n: " << i;
    for (int oc = 0; oc < output->c; oc++) {
      //std::cout << "oc: " << oc;
      for (int p = 0; p < space; p++) {
        //std::cout << "space: " << p;
        *output_feature = conv_weights->bias[oc];
        output_feature++;
      }
    }
    //output_feature += output->c;
  }
  std::cout << "CNN::convolution_layer2" << std::endl;

  output_feature = output->pixels;
  double *intput_feature = input->pixels;
  double *weight = conv_weights->pixels;  // oc, ic, kh, kw
  //int pad = conv_weights->h / 2;
  for (int occ = 0; occ < oc; occ++) {
    for (int nn = 0; nn < n; nn++) {
      intput_feature = &(input->pixels[nn * input->size]);
      //assert(intput_feature < &(input->pixels[input->n * input->size]));

      for (int ohh = 0; ohh < oh; ohh++) {
        for (int oww = 0; oww < ow; oww++) {
          output_feature = &(output->pixels[((nn * oc + occ) * oh + ohh ) * ow + oww]);
          //assert(output_feature < &(output->pixels[output->n * output->size]));

          weight = &(conv_weights->pixels[occ * ic * conv_weights->h * conv_weights->w]);
          //assert(weight < &(conv_weights->pixels[conv_weights->size]));

          int w_i = 0;
          for (int icc = 0; icc < ic; icc++) {
            for (int kh = 0; kh < conv_weights->h; kh++) {
              for (int kw = 0; kw < conv_weights->w; kw++) {
                (*output_feature) += intput_feature[(icc * ih + ohh + kh) * iw + oww + kw] * weight[w_i++];
                //assert(&(intput_feature[(icc * ih + ohh + kh) * iw + oww + kw]) < &(input->pixels[input->n * input->size]));
                //assert(&weight[w_i-1] < &(conv_weights->pixels[conv_weights->size]));
              }
            }
          }
        }
      }
    }
  }

  //print_feature(output, "convolution_layer output");
}

// conv layer definition

void conv_layer::convolution() {
  // 
  for (int  p = 0; p < feature_out->size; p++) {
    feature_out->pixels[p] = 0.0;
  }
  //
  im2col();
  matmul();
  addbias();
}

void conv_layer::im2col() {
  double* col = column;
  int pad_ = 0;
  if (!valid) {
    //pad_ = (kw - 1) / 2;
    pad_ = this->pad / 2;
  }

  int in_space = ih * iw;
  int out_space = oh * ow;
  //for (int h = -pad_; h < oh-pad_; h++) {
  //  for (int w = -pad_; w < ow-pad_; w++) {
  for (int h = -pad_, hi = 0; hi < oh; h += stride, hi++) {
    for (int w = -pad_, wi = 0; wi < ow; w += stride, wi++) {
      double *tmp = col++;
      double *im = feature_in->pixels + h * iw + w;
      for (int c = 0; c < ic; c++) { // which input channel

        for (int hh = 0; hh < kh; hh++) {
          for (int ww = 0; ww < kw; ww++) {
            *tmp = (hh + h >= 0 && ww + w >= 0 && hh + h < ih && ww + w < iw) ? *(im + hh * iw + ww): 0;
            tmp += out_space;
          }
        }
        im += in_space;
      }
    }
  }
}

void conv_layer::matmul() {
  double *output = feature_out->pixels;
  int n = oc;
  int m = oh*ow;
  int kk = ic*kh*kw;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      for (int k = 0; k < kk; k++) {
        *(output) += weight->pixels[i * kk + k] * column[k * m + j];
      }
      output++;
    }
  }
}

void conv_layer::addbias() {
  double *output = feature_out->pixels;
  for (int c = 0; c < oc; c++) {
    for (int p = 0; p < oh*ow; p++) {
      *output += weight->bias[c];
      output++;
    }
  }
}


//networks
void CNN::alexnet_flower(int h_, int w_) {
  printf("CNN::CNN\n");
  int h = h_;
  int w = w_;
  input = new feature_map(h, w, 3, 1);
  conv2d_1 = new feature_map(h, w, 32, 1);
  conv2d_layer1 = new conv_layer(h, w, 3, 32, 3, 3, 1, false, input, conv2d_1);
  activation_1 = new feature_map(h, w, 32, 1);
  h = h -2;
  w = w -2;
  conv2d_2 = new feature_map(h, w, 32, 1);
  conv2d_layer2 = new conv_layer(h+2, w+2, 32, 32, 3, 3, 1, true, activation_1, conv2d_2);
  activation_2 = new feature_map(h, w, 32, 1);
  h = h / 2;
  w = w / 2;
  max_pooling2d_1 = new feature_map(h, w, 32, 1);
  conv2d_3 = new feature_map(h, w, 64, 1);
  conv2d_layer3 = new conv_layer(h, w, 32, 64, 3, 3, 1, false, max_pooling2d_1, conv2d_3);
  activation_3 = new feature_map(h, w, 64, 1);
  h = h -2;
  w = w -2;
  conv2d_4 = new feature_map(h, w, 64, 1);
  conv2d_layer4 = new conv_layer(h+2, w+2, 64, 64, 3, 3, 1, true, activation_3, conv2d_4);
  activation_4 = new feature_map(h, w, 64, 1);
  h = h / 2;
  w = w / 2;
  max_pooling2d_2 = new feature_map(h, w, 64, 1);
  
  flatten_ = new feature_map(1, 1, h*w*64, 1);
  dense_1 = new feature_map(1, 1, 512, 1);
  dense_weights1 = new weights(1, 1, h*w*64, 512);
  activation_5 = new feature_map(1, 1, 512, 1);
  dense_2 = new feature_map(1, 1, 10, 1);
  dense_weights2 = new weights(1, 1, 512, 10);
  activation_6 = new feature_map(1, 1, 10, 1);
}

void CNN::mnist(int h_, int w_, int c_, int batch_size_) {
  printf("CNN::CNN\n");
  int h = h_;
  int w = w_;
  int c = c_;
  input = new feature_map(h, w, c, batch_size_);
  
  //flatten_ = new feature_map(1, 1, h*w*3, batch_size_);
  dense_1 = new feature_map(1, 1, 1024, batch_size_);
  dense_weights1 = new weights(1, 1, h*w*c, 1024);
  activation_5 = new feature_map(1, 1, 1024, batch_size_);
  dense_2 = new feature_map(1, 1, 10, batch_size_);
  dense_weights2 = new weights(1, 1, 1024, 10);
  activation_6 = new feature_map(1, 1, 10, batch_size_);
}


//=======================lenet start here===================================
void CNN_lenet::cnn(std::string input_path, std::string in) {
  std::cout << "CNN_lenet::cnn\n";
  lenet(input_height, input_width, input_channel, batch_size); // 1

  std::ifstream input_file;
  input_file.open(input_path + in, std::ifstream::in);

  std::string weight_filename;
  input_file >> weight_filename;
  ////load_weights(input_path + weight_filename);
  load_weights_lenet(input_path + weight_filename); // 2
  std::cout << "succ loaded weights\n";

  input_file >> num_images;
  //file >> image->w;
  input_file >> input_height;
  input_file >> input_width;
  input_file >> input_channel;

  std::string image_filename;
  int label;
  std::vector<int> predict;
  //predicts.resize()

  // inference the images one by one
  double* pixels;
  for (int i = 0; i * batch_size < num_images; i++) {
    pixels = input->pixels;
    for (int j = 0; j + i < num_images && j < batch_size; j++) {
      input_file >> image_filename;
      input_file >> label;
      labels.push_back(label);
  
      load_image_lenet(input_path + image_filename, pixels); // 3
      //std::cout << input_path << std::endl;
      //std::cout << image_filename << std::endl;
  
      std::cout << "loaded image: " << input_path + image_filename << ' ' << label << std::endl;
      pixels += input->size;
    }

    predict = inference(); // 4
    //std::cout << "predicted: " << predict << std::endl;
    //predicts.push_back(predict);
    predicts.insert(predicts.end(), predict.begin(), predict.end());
    print_feature(conv2d_22, "output conv2d_2");
    print_feature(activation_2, "output activation_2");
    //if (i>10) break;
    break;
  }

  //test_result();
}


void CNN_lenet::lenet(int h_, int w_, int c_, int batch_size_) {
  printf("CNN_lenet::lenet\n");
  int h = h_;
  int w = w_;
  int c = c_;
  int n = batch_size_;
  input = new feature_map(h, w, c, n);
  conv2d_1 = new feature_map(h-2, w-2, 512, n);  // 26 26 512 #weight1
  conv_weights1 = new weights(3, 3, 1, 512);
  h = h - 2; w = w - 2;
  activation_1 = new feature_map(h, w, 512, n);  // 26 26 512
  max_pooling2d_1 = new feature_map(h/2, w/2, 512, n);  // 13 13 512

  h = h / 2; w = w / 2;
  conv2d_22 = new feature_map(h-2, w-2, 512, n);  // 11 11 512 #weight2
  conv_weights2 = new weights(3, 3, 512, 512);
  h = h - 2; w = w - 2;
  activation_2 = new feature_map(h, w, 512, n);  // 11 11 512
  max_pooling2d_2 = new feature_map(h/2, w/2, 512, n);  // 5 5 512
  h = h / 2; w = w / 2;

  flatten_ = new feature_map(1, 1, h*w*512, n);  // 12800

  dense_1 = new feature_map(1, 1, 120, n);  // 120 #weight3
  dense_weights1 = new weights(1, 1, h*w*512, 120);
  activation_3 = new feature_map(1, 1, 120, n);  // 120

  dense_2 = new feature_map(1, 1, 84, n);  // 84 #weight4
  dense_weights2 = new weights(1, 1, 120, 84);
  activation_4 = new feature_map(1, 1, 84, n);  // 84

  dense_3 = new feature_map(1, 1, 10, n);  // 10 # weight5
  dense_weights3 = new weights(1, 1, 84, 10);
  activation_5 = new feature_map(1, 1, 10, n);  // 10
  
}


std::vector<int> CNN_lenet::inference() {

  std::cout << "CNN_lenet::inference" << std::endl;
  int *iiiii=new int(0);
  //flatten(input, flatten_);
  convolution_layer(input, conv2d_1, conv_weights1);
  relu_layer(conv2d_1, activation_1);
  maxpool_layer(activation_1, max_pooling2d_1, 2, 2);
  //for (int i = 0; i < dense_1->size; i++) {
  //  std::cout << dense_1->pixels[i] << std::endl;
  //}
  std::cout << "====================================================" << std::endl;
  //for (int i = 0; i < activation_5->size; i++) {
  //  std::cout << activation_5->pixels[i] << std::endl;
  //}
  iiiii=new int(0);
  convolution_layer(max_pooling2d_1, conv2d_22, conv_weights2);
  relu_layer(conv2d_22, activation_2);
  maxpool_layer(activation_2, max_pooling2d_2, 2, 2);

  flatten(max_pooling2d_2, flatten_);
  std::cout << "====================================================" << std::endl;
  dense_layer(flatten_, dense_1, dense_weights1, iiiii);
  relu_layer(dense_1, activation_3);
  //for (int i = 0; i < dense_2->size; i++) {
  //  std::cout << dense_2->pixels[i] << std::endl;
  //}
  std::cout << "====================================================" << std::endl;
  dense_layer(activation_3, dense_2, dense_weights2, iiiii);
  relu_layer(dense_2, activation_4);


  dense_layer(activation_4, dense_3, dense_weights3, iiiii);
  softmax_layer(dense_3, activation_5);

  //for (int i = 0; i < activation_6->size; i++) {
  //  std::cout << activation_6->pixels[i] << std::endl;
  //}

  double max;
  int catogary_res;
  std::vector<int> catogary;
  double* activation_5_pixels = activation_5->pixels;
  for (int n = 0; n < activation_5->n; n++) {
    max = 0.0;
    for(int i = 0; i < activation_5->size; i++) {
      //std::cout << *activation_5_pixels << std::endl;
      if (*activation_5_pixels > max) {
        max = *activation_5_pixels;
        catogary_res = i;
      }
      activation_5_pixels++;
    }
    //printf("%d\n", catogary_res);
    catogary.push_back(catogary_res);
  }

  return catogary;
}

void CNN_lenet::load_weights_lenet(std::string weight_filename) {
  std::cout << "CNN_lenet::load_weights_lenet" << std::endl;
  std::ifstream weight_file;
  weight_file.open(weight_filename, std::ifstream::in);

  int ic = 1;
  int oc = 512;
  int kw = 3;
  int kh = 3;

  //conv2d_layer1
  double *weights = conv_weights1->pixels;
  double *bias = conv_weights1->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }

  std::cout << "CNN_lenet::load_weights_lenet:conv2d_layer1" << std::endl;

  //conv2d_layer2
  ic = 512; oc = 512; kw = 3; kh = 3;
  weights = conv_weights2->pixels;
  bias = conv_weights2->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }
  std::cout << "CNN_lenet::load_weights_lenet:conv2d_layer2" << std::endl;

  //dense_layer1
  ic = dense_weights1->ic; oc = dense_weights1->oc; kw = 1; kh = 1;
  weights = dense_weights1->pixels;
  bias = dense_weights1->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }
  std::cout << "CNN_lenet::load_weights_lenet:dense_layer1" << std::endl;

  //dense_layer2
  ic = dense_weights2->ic; oc = dense_weights2->oc; kw = 1; kh = 1;
  weights = dense_weights2->pixels;
  bias = dense_weights2->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }
  std::cout << "CNN_lenet::load_weights_lenet:dense_layer2" << std::endl;

  //dense_layer3
  ic = dense_weights3->ic; oc = dense_weights3->oc; kw = 1; kh = 1;
  weights = dense_weights3->pixels;
  bias = dense_weights3->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
    //std::cout << weights[i] << ' ' << std::endl;
  }
  std::cout << "bias" << std::endl;
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
    //sstd::cout << bias[i] << std::endl;
  }
  std::cout << "CNN_lenet::load_weights_lenet:dense_layer3" << std::endl;

}


//=======================squeezenet start here===================================
void CNN_squeezenet::cnn(std::string input_path, std::string in) {
  std::cout << "CNN_squeezenet::cnn\n";
  squeezenet(input_height, input_width, input_channel, batch_size); // 1

  std::ifstream input_file;
  input_file.open(input_path + in, std::ifstream::in);

  std::string weight_filename;
  input_file >> weight_filename;
  ////load_weights(input_path + weight_filename);
  load_weights_squeezenet(input_path + weight_filename); // 2

  //std::cout << "1n: " << conv2d_1_e1->n << " oc: " << conv2d_1_e1->c << " space: " << conv2d_1_e1->h*conv2d_1_e1->w << std::endl;

  std::cout << "succ loaded weights\n";

  input_file >> num_images;
  //file >> image->w;
  input_file >> input_height;
  input_file >> input_width;
  input_file >> input_channel;

  std::string image_filename;
  int label;
  std::vector<int> predict;
  //predicts.resize()

  // inference the images one by one
  double* pixels;
  //std::cout << "2n: " << conv2d_1_e1->n << " oc: " << conv2d_1_e1->c << " space: " << conv2d_1_e1->h*conv2d_1_e1->w << std::endl;
  for (int i = 0; i * batch_size < num_images; i++) {
    pixels = input->pixels;
    for (int j = 0; j + i < num_images && j < batch_size; j++) {
      input_file >> image_filename;
      input_file >> label;
      labels.push_back(label);
  
      load_image_lenet(input_path + image_filename, pixels); // 3
      //std::cout << input_path << std::endl;
      //std::cout << image_filename << std::endl;
  
      std::cout << "loaded image: " << input_path + image_filename << ' ' << label << std::endl;
      pixels += input->size;
    }

    predict = inference(); // 4
    //std::cout << "predicted: " << predict << std::endl;
    //predicts.push_back(predict);
    predicts.insert(predicts.end(), predict.begin(), predict.end());
    //if (i>10) break;
    break;
  }

  //test_result();
}


void CNN_squeezenet::squeezenet(int h_, int w_, int c_, int batch_size_) {
  printf("CNN_squeezenet::squeezenet\n");
  int h = h_;
  int w = w_;
  int c = c_;
  int n = batch_size_;
  std::cout << "n: " << n << " oc: " << c << " space: " << h*w << std::endl;

  input = new feature_map(h, w, c, n);  // 28 28 1
  c = 256;
  conv2d_1_e1 = new feature_map(h, w, c, n);  // 28 28 256 #weight1
  h=h-2; w=w-2;
  crop_1 = new feature_map(h, w, c, n);  // 26 26 256
  conv2d_1_e3 = new feature_map(h, w, c, n);  // 26 26 256 #weight2
  c = c * 2;
  concat_1 = new feature_map(h, w, c, n);  // 26 26 512
  activation_1 = new feature_map(h, w, c, n);  // 26 26 512
  h=h/2; w=w/2;
  max_pooling2d_1 = new feature_map(h, w, c, n);  // 13 13 512

  c=64;
  conv2d_2_s1 = new feature_map(h, w, c, n);  // 13 13 64 #weight3
  activation_2 = new feature_map(h, w, c, n);  // 26 26 512
  c=256;
  conv2d_2_e1 = new feature_map(h, w, c, n);  // 13 13 256 #weight4
  h=h-2; w=w-2;
  crop_2 = new feature_map(h, w, c, n);  // 11 11 256
  conv2d_2_e3 = new feature_map(h, w, c, n);  // 11 11 256 #weight5
  c=c*2;
  concat_2 = new feature_map(h, w, c, n);  // 11 11 512
  activation_3 = new feature_map(h, w, c, n);  // 11 11 512
  h=h/2; w=w/2;
  max_pooling2d_2 = new feature_map(h, w, c, n);  // 5 5 512

  c=h*w*c; h=1; w=1;
  flatten_ = new feature_map(h, w, c, n);  // 12800
  dense_weights1 = new weights(1, 1, c, 120);
  c=120;
  dense_1 = new feature_map(h, w, c, n);  // 120 #weight3
  activation_4 = new feature_map(h, w, c, n);  // 120

  dense_weights2 = new weights(1, 1, 120, 84);
  c=84;
  dense_2 = new feature_map(h, w, c, n);  // 84 #weight4
  activation_5 = new feature_map(h, w, c, n);  // 84

  dense_weights3 = new weights(1, 1, 84, 10);
  c=10;
  dense_3 = new feature_map(h, w, c, n);  // 10 # weight5
  activation_6 = new feature_map(h, w, c, n);  // 10

  // conv_layer weights:
  conv_weights1 = new weights(1, 1, 1, 256);
  conv_weights2 = new weights(3, 3, 1, 256);
  conv_weights3 = new weights(1, 1, 512, 64);
  conv_weights4 = new weights(1, 1, 64, 256);
  conv_weights5 = new weights(3, 3, 64, 256);
  
}


std::vector<int> CNN_squeezenet::inference() {

  std::cout << "CNN_squeezenet::inference" << std::endl;
  std::cout << "n: " << conv2d_1_e1->n << " oc: " << conv2d_1_e1->c << " space: " << conv2d_1_e1->h*conv2d_1_e1->w << std::endl;
  int *iiiii=new int(0);
  //flatten(input, flatten_);
  convolution_layer(input, conv2d_1_e1, conv_weights1);
  convolution_layer(input, conv2d_1_e3, conv_weights2);
  crop_layer(conv2d_1_e1, crop_1, 1);
  concat_layer(crop_1, conv2d_1_e3, concat_1);
  relu_layer(concat_1, activation_1);
  maxpool_layer(activation_1, max_pooling2d_1, 2, 2);
  //for (int i = 0; i < dense_1->size; i++) {
  //  std::cout << dense_1->pixels[i] << std::endl;
  //}
  std::cout << "====================================================" << std::endl;
  //for (int i = 0; i < activation_5->size; i++) {
  //  std::cout << activation_5->pixels[i] << std::endl;
  //}
  iiiii=new int(0);
  convolution_layer(max_pooling2d_1, conv2d_2_s1, conv_weights3);
  relu_layer(conv2d_2_s1, activation_2);
  convolution_layer(activation_2, conv2d_2_e1, conv_weights4);
  convolution_layer(activation_2, conv2d_2_e3, conv_weights5);
  crop_layer(conv2d_2_e1, crop_2, 1);
  concat_layer(crop_2, conv2d_2_e3, concat_2);
  relu_layer(concat_2, activation_3);
  maxpool_layer(activation_3, max_pooling2d_2, 2, 2);


  flatten(max_pooling2d_2, flatten_);
  std::cout << "====================================================" << std::endl;
  dense_layer(flatten_, dense_1, dense_weights1, iiiii);
  relu_layer(dense_1, activation_4);
  //for (int i = 0; i < dense_2->size; i++) {
  //  std::cout << dense_2->pixels[i] << std::endl;
  //}
  std::cout << "====================================================" << std::endl;
  dense_layer(activation_4, dense_2, dense_weights2, iiiii);
  relu_layer(dense_2, activation_5);


  dense_layer(activation_5, dense_3, dense_weights3, iiiii);
  softmax_layer(dense_3, activation_6);

  //for (int i = 0; i < activation_6->size; i++) {
  //  std::cout << activation_6->pixels[i] << std::endl;
  //}

  double max;
  int catogary_res;
  std::vector<int> catogary;
  double* activation_6_pixels = activation_6->pixels;
  for (int n = 0; n < activation_6->n; n++) {
    max = 0.0;
    for(int i = 0; i < activation_6->size; i++) {
      //std::cout << *activation_6_pixels << std::endl;
      if (*activation_6_pixels > max) {
        max = *activation_6_pixels;
        catogary_res = i;
      }
      activation_6_pixels++;
    }
    //printf("%d\n", catogary_res);
    catogary.push_back(catogary_res);
  }

  return catogary;
}

void CNN_squeezenet::load_weights_squeezenet(std::string weight_filename) {
  std::cout << "CNN_squeezenet::load_weights_squeezenet " << weight_filename << std::endl;
  std::ifstream weight_file;
  weight_file.open(weight_filename, std::ifstream::in);

  int ic = 1;
  int oc = 256;
  int kw = 1;
  int kh = 1;

  //conv2d_layer1_e1
  double *weights = conv_weights1->pixels;
  double *bias = conv_weights1->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }

  std::cout << "CNN_squeezenet::load_weights_squeezenet:conv2d_layer1_e1" << std::endl;

  //conv2d_layer1_e3
  ic = conv_weights2->ic; oc = conv_weights2->oc; kw = 3; kh = 3;
  weights = conv_weights2->pixels;
  bias = conv_weights2->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }
  std::cout << "CNN_squeezenet::load_weights_squeezenet:conv2d_layer1_e3" << std::endl;

  //conv2d_layer2_s1
  ic = conv_weights3->ic; oc = conv_weights3->oc; kw = 1; kh = 1;
  weights = conv_weights3->pixels;
  bias = conv_weights3->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }
  std::cout << "CNN_squeezenet::load_weights_squeezenet:conv2d_layer2_s1" << std::endl;

  //conv2d_layer2_e1
  ic = conv_weights4->ic; oc = conv_weights4->oc; kw = 1; kh = 1;
  weights = conv_weights4->pixels;
  bias = conv_weights4->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }
  std::cout << "CNN_squeezenet::load_weights_squeezenet:conv2d_layer2_e1" << std::endl;

  //conv2d_layer2_e3
  ic = conv_weights5->ic; oc = conv_weights5->oc; kw = 3; kh = 3;
  weights = conv_weights5->pixels;
  bias = conv_weights5->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }
  std::cout << "CNN_squeezenet::load_weights_squeezenet:conv2d_layer2_e3" << std::endl;

  //dense_layer1
  ic = dense_weights1->ic; oc = dense_weights1->oc; kw = 1; kh = 1;
  weights = dense_weights1->pixels;
  bias = dense_weights1->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }
  std::cout << "CNN_squeezenet::load_weights_squeezenet:dense_layer1" << std::endl;

  //dense_layer2
  ic = dense_weights2->ic; oc = dense_weights2->oc; kw = 1; kh = 1;
  weights = dense_weights2->pixels;
  bias = dense_weights2->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
  }
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
  }
  std::cout << "CNN_squeezenet::load_weights_squeezenet:dense_layer2" << std::endl;

  //dense_layer3
  ic = dense_weights3->ic; oc = dense_weights3->oc; kw = 1; kh = 1;
  weights = dense_weights3->pixels;
  bias = dense_weights3->bias;
  for (int i = 0; i < ic*oc*kw*kh; i++) {
    weight_file >> weights[i];
    //std::cout << weights[i] << ' ' << std::endl;
  }
  std::cout << "bias" << std::endl;
  for (int i = 0; i < oc; i++) {
    weight_file >> bias[i];
    //sstd::cout << bias[i] << std::endl;
  }
  std::cout << "CNN_squeezenet::load_weights_squeezenet:dense_layer3" << std::endl;

}


// functions for Fluid:

void CNN::relu_layer_E(feature_map* input, feature_map* output, int* iiiii) {
  std::cout << "CNN::relu_layer_E start" << std::endl;
  relu_layer(input, output);
  *iiiii = 2;
  std::cout << "CNN::relu_layer_E end" << std::endl;
  //print_feature(input, "relu_layer_E input");
  //print_feature(output, "relu_layer_E output");
}


void CNN::softmax_layer_E(feature_map* input, feature_map* output, int* iiiii) {
  std::cout << "CNN::softmax_layer_E start" << std::endl;
  softmax_layer(input, output);
  *iiiii = 2;
  std::cout << "CNN::softmax_layer_E end" << std::endl;
  //print_feature(output, "softmax_layer_E");
}


void CNN::maxpool_layer_E(feature_map* input, feature_map* output, int k, int stride, int* iiiii) {
  std::cout << "CNN::maxpool_layer_E start" << std::endl;
  maxpool_layer(input, output, k, stride);
  *iiiii = 2;
  std::cout << "CNN::maxpool_layer_E end" << std::endl;
  //print_feature(output, "maxpool_layer_E");
}


void CNN::dense_layer_E(feature_map* input, feature_map* output, weights* dense_weights, double scale, int *iiiii) {
  std::cout << "CNN::dense_layer_E start" << std::endl;
  //struct timeval t0, t1;
  //gettimeofday(&t0, 0);

  assert(input->c == dense_weights->ic);
  assert(output->c == dense_weights->oc);
  assert(input->h == 1);
  assert(output->h == 1);
  assert(dense_weights->h == 1);
  assert(input->w == 1);
  assert(output->w == 1);
  assert(dense_weights->w == 1);
  assert(input->n == output->n);

  double *output_feature = output->pixels;
  for (int i = 0; i < output->n; i++) {
    for (int oc = 0; oc < output->c; oc++) {
      output_feature[oc] = dense_weights->bias[oc];
    }
    output_feature += output->c;
  }
  std::cout << "dense weights: " << std::endl;
  //gettimeofday(&t1, 0);
  //long elapsed = (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;
  std::cout << "CNN::dense_layer_E end " << std::endl;

  double *weights = dense_weights->pixels;
  double *input_feature = input->pixels;
  for (*iiiii = 0; *iiiii < output->c; (*iiiii)++) {
    //std::cout << *iiiii << std::endl;
    for (int ic = 0; ic < input->c; ic++) {
      input_feature = input->pixels + ic;
      output_feature = output->pixels + (*iiiii);
      for (int i = 0; i < output->n; i++) {
        //std::cout << output_feature << ' ' << input_feature << ' ' << i << std::endl;
        *output_feature += *input_feature * (*weights) * scale;
        input_feature += input->c;
        output_feature += output->c;
        //std::cout << *weights << std::endl;
      }
      
      weights++;
    }
  }
  //print_feature(output, "dense_layer_E");
}


void CNN::flatten_E(feature_map* input, feature_map* output, int* iiiii) {
  std::cout << "CNN::flatten_E start" << std::endl;
  flatten(input, output);
  *iiiii = 2;
  std::cout << "CNN::flatten_E end" << std::endl;
  //print_feature(output, "flatten_E");
}


void CNN::convolution_layer_E(feature_map* input, feature_map* output, weights* conv_weights, double scale, int* iiiii) {
  int n = input->n;
  int ic = input->c;
  int ih = input->h;
  int iw = input->w;
  int oc = output->c;
  int oh = output->h;
  int ow = output->w;

  std::cout << "CNN::convolution_layer_E start" << conv_weights->ic << ' ' << conv_weights->oc << ' ' << conv_weights->h << ' ' << conv_weights->w << std::endl;
  double *output_feature = output->pixels;
  int space = output->h * output->w;
  for (int i = 0; i < output->n; i++) {
    for (int oc = 0; oc < output->c; oc++) {
      for (int p = 0; p < space; p++) {
        *output_feature = conv_weights->bias[oc];
        output_feature++;
      }
    }
    //output_feature += output->c;
  }
  std::cout << "CNN::convolution_layer_E mid" << conv_weights->ic << ' ' << conv_weights->oc << ' ' << conv_weights->h << ' ' << conv_weights->w << std::endl;

  output_feature = output->pixels;
  double *intput_feature = input->pixels;
  double *weight = conv_weights->pixels;  // oc, ic, kh, kw occ
  //int pad = conv_weights->h / 2;
  //for (int occ = 0; occ < oc; occ++) {
  for ((*iiiii) = 0; (*iiiii) < oc; (*iiiii)++) {
    //printf("%d\n", (*iiiii));
    for (int nn = 0; nn < n; nn++) {
      intput_feature = &(input->pixels[nn * input->size]);
      //assert(intput_feature < &(input->pixels[input->n * input->size]));

      for (int ohh = 0; ohh < oh; ohh++) {
        for (int oww = 0; oww < ow; oww++) {
          output_feature = &(output->pixels[((nn * oc + (*iiiii)) * oh + ohh ) * ow + oww]);
          //assert(output_feature < &(output->pixels[output->n * output->size]));

          weight = &(conv_weights->pixels[(*iiiii) * ic * conv_weights->h * conv_weights->w]);
          //assert(weight < &(conv_weights->pixels[conv_weights->size]));

          int w_i = 0;
          for (int icc = 0; icc < ic; icc++) {
            for (int kh = 0; kh < conv_weights->h; kh++) {
              for (int kw = 0; kw < conv_weights->w; kw++) {
                (*output_feature) += intput_feature[(icc * ih + ohh + kh) * iw + oww + kw] * weight[w_i++] * scale;
                //assert(&(intput_feature[(icc * ih + ohh + kh) * iw + oww + kw]) < &(input->pixels[input->n * input->size]));
                //assert(&weight[w_i-1] < &(conv_weights->pixels[conv_weights->size]));
              }
            }
          }
        }
      }
    }
  }
  std::cout << "CNN::convolution_layer_E end" << conv_weights->ic << ' ' << conv_weights->oc << ' ' << conv_weights->h << ' ' << conv_weights->w << std::endl;
  //print_feature(output, "convolution_layer_E");
}


void CNN::crop_layer_E(feature_map* input, feature_map* output, int cut, int* iiiii) {
  std::cout << "CNN::crop_layer_E start" << std::endl;
  crop_layer(input, output, cut);
  *iiiii = 2;
  std::cout << "CNN::crop_layer_E end" << std::endl;
  //print_feature(output, "crop_layer_E");
}


void CNN::concat_layer_E(feature_map* input1, feature_map* input2, feature_map* output, int* iiiii) {
  std::cout << "CNN::concat_layer_E start" << std::endl;
  concat_layer(input1, input2, output);
  *iiiii = 2;
  std::cout << "CNN::concat_layer_E end" << std::endl;
  //print_feature(output, "concat_layer_E");
}
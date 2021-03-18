#include "cnn.h"
#include "cnn_fluid.h"

using namespace example;

//=================================lenet start  here: =========================================

std::vector<int> CNN_lenetFluid_E::inference() {

  std::cout << "CNN_lenetFluid_E::inference" << std::endl;

  auto tf = new TaskFactory();
  auto gs = new AggressiveGS(80);
  auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
  auto tp0status = new __fluid__<bool>(tp0->isfinished());
  auto v0 = new ValveEQ<bool>(tp0status, true);
  tp0->run();

  Data *d_input = new Data;
  d_input->set_data((void*)input);
  Data *d_conv2d_1 = new Data;
  d_conv2d_1->set_data((void*)conv2d_1);
  Data *d_activation_1 = new Data;
  d_activation_1->set_data((void*)activation_1);
  Data *d_max_pooling2d_1 = new Data;
  d_max_pooling2d_1->set_data((void*)max_pooling2d_1);
  Data *d_conv2d_2 = new Data;
  d_conv2d_2->set_data((void*)NULL);
  Data *d_activation_2 = new Data;
  d_activation_2->set_data((void*)activation_2);
  Data *d_max_pooling2d_2 = new Data;
  d_max_pooling2d_2->set_data((void*)max_pooling2d_2);

  Data *d_flatten_ = new Data;
  d_flatten_->set_data((void*)flatten_);
  Data *d_dense_1 = new Data;
  d_dense_1->set_data((void*)dense_1);
  Data *d_activation_3 = new Data;
  d_activation_3->set_data((void*)activation_3);
  Data *d_dense_2 = new Data;
  d_dense_2->set_data((void*)dense_2);
  Data *d_activation_4 = new Data;
  d_activation_4->set_data((void*)activation_4);
  Data *d_dense_3 = new Data;
  d_dense_3->set_data((void*)dense_3);
  Data *d_activation_5 = new Data;
  d_activation_5->set_data((void*)activation_5);


  ////#pragma call_num call_num_conv2d_1;
  auto call_num_conv2d_1 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g1,{},{},{d_input},{d_conv2d_1}>>>CNN_lenetFluid_E::convolution_layer_E(input, conv2d_1, conv_weights1, 1.0, call_num_conv2d_1->p);
  auto tpb__g1 = std::bind(&CNN_lenetFluid_E::convolution_layer_E, this, input, conv2d_1, conv_weights1, 1.0, call_num_conv2d_1->p);
  auto tp__g1 = tf->newTask<decltype(tpb__g1)>("conv_layer" + std::to_string(0), {d_input}, {d_conv2d_1}, tpb__g1);
  Guard* g1 = gs->newGuard("conv_layer" + std::to_string(0), {}, {}, tp__g1, {});
  guard_log.push_back(g1);
  ////#pragma valve v_conv2d_1 = v1.init(call_num_conv2d_1, conv2d_1->c*rate);
  auto v_conv2d_1 = v1.init(call_num_conv2d_1, conv2d_1->c*rate);
  ////g1->set_root;
  g1->set_root();


  ////#pragma call_num call_num_activation_1;
  auto call_num_activation_1 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g2,{v_conv2d_1},{},{d_conv2d_1},{d_activation_1}>>>CNN_lenetFluid_E::relu_layer_E(conv2d_1, activation_1, call_num_activation_1->p);
  auto tpb__g2 = std::bind(&CNN_lenetFluid_E::relu_layer_E, this, conv2d_1, activation_1, call_num_activation_1->p);
  auto tp__g2 = tf->newTask<decltype(tpb__g2)>("relu_layer" + std::to_string(0), {d_conv2d_1}, {d_activation_1}, tpb__g2);
  Guard* g2 = gs->newGuard("relu_layer" + std::to_string(0), {v_conv2d_1}, {}, tp__g2, {});
  guard_log.push_back(g2);
  ////#pragma valve v_activation_1 = v1.init(call_num_activation_1, 1);
  auto v_activation_1 = v1.init(call_num_activation_1, 1);


  ////#pragma call_num call_num_max_pooling2d_1;
  auto call_num_max_pooling2d_1 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g3,{v_activation_1},{},{d_activation_1},{d_max_pooling2d_1}>>>CNN_lenetFluid_E::maxpool_layer_E(activation_1, max_pooling2d_1, 2, 2, call_num_max_pooling2d_1->p);
  auto tpb__g3 = std::bind(&CNN_lenetFluid_E::maxpool_layer_E, this, activation_1, max_pooling2d_1, 2, 2, call_num_max_pooling2d_1->p);
  auto tp__g3 = tf->newTask<decltype(tpb__g3)>("maxpool_layer" + std::to_string(0), {d_activation_1}, {d_max_pooling2d_1}, tpb__g3);
  Guard* g3 = gs->newGuard("maxpool_layer" + std::to_string(0), {v_activation_1}, {}, tp__g3, {});
  guard_log.push_back(g3);
  ////#pragma valve v_max_pooling2d_1 = v1.init(call_num_max_pooling2d_1, 1);
  auto v_max_pooling2d_1 = v1.init(call_num_max_pooling2d_1, 1);


  ////#pragma call_num call_num_conv2d_2;
  auto call_num_conv2d_2 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g4,{v_max_pooling2d_1},{},{d_max_pooling2d_1},{d_conv2d_2}>>>CNN_lenetFluid_E::convolution_layer_E(max_pooling2d_1, conv2d_2, conv_weights2, 1.0 / rate, call_num_conv2d_2->p);
  auto tpb__g4 = std::bind(&CNN_lenetFluid_E::convolution_layer_E, this, max_pooling2d_1, conv2d_22, conv_weights2, 1.0 / rate, call_num_conv2d_2->p);
  auto tp__g4 = tf->newTask<decltype(tpb__g4)>("conv_layer" + std::to_string(1), {d_max_pooling2d_1}, {d_conv2d_2}, tpb__g4);
  Guard* g4 = gs->newGuard("conv_layer" + std::to_string(1), {v_max_pooling2d_1}, {}, tp__g4, {});
  guard_log.push_back(g4);
  ////#pragma valve v_conv2d_2 = v1.init(call_num_conv2d_2, conv2d_2->c*rate);
  auto v_conv2d_2 = v1.init(call_num_conv2d_2, conv2d_22->c*rate);


  ////#pragma call_num call_num_activation_2;
  auto call_num_activation_2 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g5,{v_conv2d_2},{},{d_conv2d_2},{d_activation_2}>>>CNN_lenetFluid_E::relu_layer_E(conv2d_2, activation_2, call_num_activation_2->p);
  auto tpb__g5 = std::bind(&CNN_lenetFluid_E::relu_layer_E, this, conv2d_22, activation_2, call_num_activation_2->p);
  auto tp__g5 = tf->newTask<decltype(tpb__g5)>("relu_layer" + std::to_string(1), {d_conv2d_2}, {d_activation_2}, tpb__g5);
  Guard* g5 = gs->newGuard("relu_layer" + std::to_string(1), {v_conv2d_2}, {}, tp__g5, {});
  guard_log.push_back(g5);
  ////#pragma valve v_activation_2 = v1.init(call_num_activation_2, 1);
  auto v_activation_2 = v1.init(call_num_activation_2, 1);


  ////#pragma call_num call_num_max_pooling2d_2;
  auto call_num_max_pooling2d_2 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g6,{v_activation_2},{},{d_activation_2},{d_max_pooling2d_2}>>>CNN_lenetFluid_E::maxpool_layer_E(activation_2, max_pooling2d_2, 2, 2, call_num_max_pooling2d_2->p);
  auto tpb__g6 = std::bind(&CNN_lenetFluid_E::maxpool_layer_E, this, activation_2, max_pooling2d_2, 2, 2, call_num_max_pooling2d_2->p);
  auto tp__g6 = tf->newTask<decltype(tpb__g6)>("maxpool_layer" + std::to_string(1), {d_activation_2}, {d_max_pooling2d_2}, tpb__g6);
  Guard* g6 = gs->newGuard("maxpool_layer" + std::to_string(1), {v_activation_2}, {}, tp__g6, {});
  guard_log.push_back(g6);
  ////#pragma valve v_max_pooling2d_2 = v1.init(call_num_max_pooling2d_2, 1);
  auto v_max_pooling2d_2 = v1.init(call_num_max_pooling2d_2, 1);


  ////#pragma call_num call_num_flatten_;
  auto call_num_flatten_ = new __fluid__<int>(new int(0));
  ////#pragma task <<<g7,{v_max_pooling2d_2},{},{d_max_pooling2d_2},{d_flatten_}>>>CNN_lenetFluid_E::flatten_E(max_pooling2d_2, flatten_, call_num_flatten_->p);
  auto tpb__g7 = std::bind(&CNN_lenetFluid_E::flatten_E, this, max_pooling2d_2, flatten_, call_num_flatten_->p);
  auto tp__g7 = tf->newTask<decltype(tpb__g7)>("flatten" + std::to_string(0), {d_max_pooling2d_2}, {d_flatten_}, tpb__g7);
  Guard* g7 = gs->newGuard("flatten" + std::to_string(0), {v_max_pooling2d_2}, {}, tp__g7, {});
  guard_log.push_back(g7);
  ////#pragma valve v_flatten_ = v1.init(call_num_flatten_, 1);
  auto v_flatten_ = v1.init(call_num_flatten_, 1);


  std::cout << "====================================================" << std::endl;


  ////#pragma call_num call_num_dense_1;
  auto call_num_dense_1 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g8,{v_flatten_},{},{d_flatten_},{d_dense_1}>>>CNN_lenetFluid_E::dense_layer_E(flatten_, dense_1, dense_weights1, 1.0 / rate, call_num_dense_1->p);
  auto tpb__g8 = std::bind(&CNN_lenetFluid_E::dense_layer_E, this, flatten_, dense_1, dense_weights1, 1.0 / rate, call_num_dense_1->p);
  auto tp__g8 = tf->newTask<decltype(tpb__g8)>("dense_layer" + std::to_string(0), {d_flatten_}, {d_dense_1}, tpb__g8);
  Guard* g8 = gs->newGuard("dense_layer" + std::to_string(0), {v_flatten_}, {}, tp__g8, {});
  guard_log.push_back(g8);
  ////#pragma valve v_dense_1 = v1.init(call_num_dense_1, 1);
  auto v_dense_1 = v1.init(call_num_dense_1, dense_1->c * 1);


  ////#pragma call_num call_num_activation_3;
  auto call_num_activation_3 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g9,{v_dense_1},{},{d_dense_1},{d_activation_3}>>>CNN_lenetFluid_E::relu_layer_E(dense_1, activation_3, call_num_activation_3->p);
  auto tpb__g9 = std::bind(&CNN_lenetFluid_E::relu_layer_E, this, dense_1, activation_3, call_num_activation_3->p);
  auto tp__g9 = tf->newTask<decltype(tpb__g9)>("relu_layer" + std::to_string(2), {d_dense_1}, {d_activation_3}, tpb__g9);
  Guard* g9 = gs->newGuard("relu_layer" + std::to_string(2), {v_dense_1}, {}, tp__g9, {});
  guard_log.push_back(g9);
  ////#pragma valve v_activation_3 = v1.init(call_num_activation_3, 1);
  auto v_activation_3 = v1.init(call_num_activation_3, 1);


  std::cout << "====================================================" << std::endl;


  ////#pragma call_num call_num_call_num_dense_2;
  auto call_num_call_num_dense_2 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g10,{v_activation_3},{},{d_activation_3},{d_dense_2}>>>CNN_lenetFluid_E::dense_layer_E(activation_3, dense_2, dense_weights2, 1.0, call_num_call_num_dense_2->p);
  auto tpb__g10 = std::bind(&CNN_lenetFluid_E::dense_layer_E, this, activation_3, dense_2, dense_weights2, 1.0, call_num_call_num_dense_2->p);
  auto tp__g10 = tf->newTask<decltype(tpb__g10)>("dense_layer" + std::to_string(1), {d_activation_3}, {d_dense_2}, tpb__g10);
  Guard* g10 = gs->newGuard("dense_layer" + std::to_string(1), {v_activation_3}, {}, tp__g10, {});
  guard_log.push_back(g10);
  ////#pragma valve v_dense_2 = v1.init(call_num_call_num_dense_2, 1);
  auto v_dense_2 = v1.init(call_num_call_num_dense_2, dense_2->c * 1);


  ////#pragma call_num call_num_activation_4;
  auto call_num_activation_4 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g11,{v_dense_2},{},{d_dense_2},{d_activation_4}>>>CNN_lenetFluid_E::relu_layer_E(dense_2, activation_4, call_num_activation_4->p);
  auto tpb__g11 = std::bind(&CNN_lenetFluid_E::relu_layer_E, this, dense_2, activation_4, call_num_activation_4->p);
  auto tp__g11 = tf->newTask<decltype(tpb__g11)>("relu_layer" + std::to_string(3), {d_dense_2}, {d_activation_4}, tpb__g11);
  Guard* g11 = gs->newGuard("relu_layer" + std::to_string(3), {v_dense_2}, {}, tp__g11, {});
  guard_log.push_back(g11);
  ////#pragma valve v_activation_4 = v1.init(call_num_activation_4, 1);
  auto v_activation_4 = v1.init(call_num_activation_4, 1);


  ////#pragma call_num call_num_call_num_dense_3;
  auto call_num_call_num_dense_3 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g12,{v_activation_4},{},{d_activation_4},{d_dense_3}>>>CNN_lenetFluid_E::dense_layer_E(activation_4, dense_3, dense_weights3, 1.0, call_num_call_num_dense_3->p);
  auto tpb__g12 = std::bind(&CNN_lenetFluid_E::dense_layer_E, this, activation_4, dense_3, dense_weights3, 1.0, call_num_call_num_dense_3->p);
  auto tp__g12 = tf->newTask<decltype(tpb__g12)>("dense_layer" + std::to_string(2), {d_activation_4}, {d_dense_3}, tpb__g12);
  Guard* g12 = gs->newGuard("dense_layer" + std::to_string(2), {v_activation_4}, {}, tp__g12, {});
  guard_log.push_back(g12);
  ////#pragma valve v_dense_3 = v1.init(call_num_call_num_dense_3, 1);
  auto v_dense_3 = v1.init(call_num_call_num_dense_3, dense_3->c * 1);


  ////#pragma call_num call_num_activation_5;
  auto call_num_activation_5 = new __fluid__<int>(new int(0));
  ////#pragma valve v_activation_5 = v1.init(call_num_activation_5, 1);
  auto v_activation_5 = v1.init(call_num_activation_5, 1);
  ////#pragma task <<<g13,{v_dense_3},{},{d_dense_3},{d_activation_5}>>>CNN_lenetFluid_E::softmax_layer_E(dense_3, activation_5, call_num_activation_5->p);
  auto tpb__g13 = std::bind(&CNN_lenetFluid_E::softmax_layer_E, this, dense_3, activation_5, call_num_activation_5->p);
  auto tp__g13 = tf->newTask<decltype(tpb__g13)>("softmax_layer" + std::to_string(0), {d_dense_3}, {d_activation_5}, tpb__g13);
  Guard* g13 = gs->newGuard("softmax_layer" + std::to_string(0), {v_dense_3}, {v_activation_5}, tp__g13, {});
  guard_log.push_back(g13);
  /////g13->set_leaf();
  g13->set_leaf();


  gs->synctask(tp__g13);

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

//=================================squeezenet start here:======================================

std::vector<int> CNN_squeezenetFluid_E::inference() {

  std::cout << "CNN_squeezenetFluid_E::inference" << std::endl;

  auto tf = new TaskFactory();
  auto gs = new AggressiveGS(80);
  auto tp0 = tf->newTask<void(*)(void *), void *>("Begin", &nullfun, NULL);
  auto tp0status = new __fluid__<bool>(tp0->isfinished());
  auto v0 = new ValveEQ<bool>(tp0status, true);
  tp0->run();

  Data *d_input = new Data;
  d_input->set_data((void*)input);
  Data *d_input2 = new Data;
  d_input2->set_data((void*)input);
  Data *d_conv2d_1_e1 = new Data;
  d_conv2d_1_e1->set_data((void*)conv2d_1_e1);
  Data *d_crop_1 = new Data;
  d_crop_1->set_data((void*)crop_1);
  Data *d_conv2d_1_e3 = new Data;
  d_conv2d_1_e3->set_data((void*)conv2d_1_e3);
  Data *d_concat_1 = new Data;
  d_concat_1->set_data((void*)concat_1);
  Data *d_activation_1 = new Data;
  d_activation_1->set_data((void*)activation_1);
  Data *d_max_pooling2d_1 = new Data;
  d_max_pooling2d_1->set_data((void*)max_pooling2d_1);


  Data *d_conv2d_2_s1 = new Data;
  d_conv2d_2_s1->set_data((void*)conv2d_2_s1);
  Data *d_activation_2 = new Data;
  d_activation_2->set_data((void*)activation_2);
  Data *d_activation_22 = new Data;
  d_activation_22->set_data((void*)activation_2);
  Data *d_conv2d_2_e1 = new Data;
  d_conv2d_2_e1->set_data((void*)conv2d_2_e1);
  Data *d_crop_2 = new Data;
  d_crop_2->set_data((void*)crop_2);
  Data *d_conv2d_2_e3 = new Data;
  d_conv2d_2_e3->set_data((void*)conv2d_2_e3);
  Data *d_concat_2 = new Data;
  d_concat_2->set_data((void*)concat_2);
  Data *d_activation_3 = new Data;
  d_activation_3->set_data((void*)activation_3);
  Data *d_max_pooling2d_2 = new Data;
  d_max_pooling2d_2->set_data((void*)max_pooling2d_2);


  Data *d_flatten_ = new Data;
  d_flatten_->set_data((void*)flatten_);
  Data *d_dense_1 = new Data;
  d_dense_1->set_data((void*)dense_1);
  Data *d_activation_4 = new Data;
  d_activation_4->set_data((void*)activation_4);
  Data *d_dense_2 = new Data;
  d_dense_2->set_data((void*)dense_2);
  Data *d_activation_5 = new Data;
  d_activation_5->set_data((void*)activation_5);
  Data *d_dense_3 = new Data;
  d_dense_3->set_data((void*)dense_3);
  Data *d_activation_6 = new Data;
  d_activation_6->set_data((void*)activation_6);


  ////#pragma call_num call_num_conv2d_1_e1;
  auto call_num_conv2d_1_e1 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g1,{},{},{d_input},{d_conv2d_1_e1}>>>CNN_squeezenetFluid_E::convolution_layer_E(input, conv2d_1_e1, conv_weights1, 1.0 / rate, call_num_conv2d_1_e1->p);
  auto tpb__g1 = std::bind(&CNN_squeezenetFluid_E::convolution_layer_E, this, input, conv2d_1_e1, conv_weights1, 1.0 , call_num_conv2d_1_e1->p);
  auto tp__g1 = tf->newTask<decltype(tpb__g1)>("conv_layer" + std::to_string(0), {d_input}, {d_conv2d_1_e1}, tpb__g1);
  Guard* g1 = gs->newGuard("conv_layer" + std::to_string(0), {}, {}, tp__g1, {});
  guard_log.push_back(g1);
  ////g1->set_root;
  g1->set_root();
  ////#pragma valve v_conv2d_1_e1 = v1.init(call_num_conv2d_1_e1, conv2d_1_e1->c*rate);
  auto v_conv2d_1_e1 = v1.init(call_num_conv2d_1_e1, conv2d_1_e1->c*rate);


  ////#pragma call_num call_num_conv2d_1_e3;
  auto call_num_conv2d_1_e3 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g2,{},{},{},{d_conv2d_1_e3}>>>CNN_squeezenetFluid_E::convolution_layer_E(input, conv2d_1_e3, conv_weights2, 1.0 / rate, call_num_conv2d_1_e3->p);
  auto tpb__g2 = std::bind(&CNN_squeezenetFluid_E::convolution_layer_E, this, input, conv2d_1_e3, conv_weights2, 1.0 , call_num_conv2d_1_e3->p);
  auto tp__g2 = tf->newTask<decltype(tpb__g2)>("conv_layer" + std::to_string(1), {}, {d_conv2d_1_e3}, tpb__g2);
  Guard* g2 = gs->newGuard("conv_layer" + std::to_string(1), {}, {}, tp__g2, {});
  guard_log.push_back(g2);
  ////#pragma valve v_conv2d_1_e3 = v1.init(call_num_conv2d_1_e3, conv2d_1_e3->c*rate);
  auto v_conv2d_1_e3 = v1.init(call_num_conv2d_1_e3, conv2d_1_e3->c*1);


  ////#pragma call_num call_num_crop_1;
  auto call_num_crop_1 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g3,{v_conv2d_1_e1},{},{d_conv2d_1_e1},{d_crop_1}>>>CNN_squeezenetFluid_E::crop_layer_E(conv2d_1_e1, crop_1, 1, call_num_crop_1->p);
  auto tpb__g3 = std::bind(&CNN_squeezenetFluid_E::crop_layer_E, this, conv2d_1_e1, crop_1, 1, call_num_crop_1->p);
  auto tp__g3 = tf->newTask<decltype(tpb__g3)>("crop_layer" + std::to_string(0), {d_conv2d_1_e1}, {d_crop_1}, tpb__g3);
  Guard* g3 = gs->newGuard("crop_layer" + std::to_string(0), {v_conv2d_1_e1}, {}, tp__g3, {});
  guard_log.push_back(g3);
  ////#pragma valve v_crop_1 = v1.init(call_num_crop_1, 1);
  auto v_crop_1 = v1.init(call_num_crop_1, 1);


  ////#pragma call_num call_num_concat_1;
  auto call_num_concat_1 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g4,{v_crop_1},{},{d_crop_1, d_conv2d_1_e3},{d_concat_1}>>>CNN_squeezenetFluid_E::concat_layer_E(crop_1, conv2d_1_e3, concat_1, call_num_concat_1->p);
  auto tpb__g4 = std::bind(&CNN_squeezenetFluid_E::concat_layer_E, this, crop_1, conv2d_1_e3, concat_1, call_num_concat_1->p);
  auto tp__g4 = tf->newTask<decltype(tpb__g4)>("concat_layer" + std::to_string(0), {d_crop_1, d_conv2d_1_e3}, {d_concat_1}, tpb__g4);
  Guard* g4 = gs->newGuard("concat_layer" + std::to_string(0), {v_crop_1, v_conv2d_1_e3}, {}, tp__g4, {});
  guard_log.push_back(g4);
  ////#pragma valve v_concat_1 = v1.init(call_num_concat_1, 1);
  auto v_concat_1 = v1.init(call_num_concat_1, 1);


  ////#pragma call_num call_num_activation_1;
  auto call_num_activation_1 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g5,{v_concat_1},{},{d_concat_1},{d_activation_1}>>>CNN_squeezenetFluid_E::relu_layer_E(concat_1, activation_1, call_num_activation_1->p);
  auto tpb__g5 = std::bind(&CNN_squeezenetFluid_E::relu_layer_E, this, concat_1, activation_1, call_num_activation_1->p);
  auto tp__g5 = tf->newTask<decltype(tpb__g5)>("relu_layer" + std::to_string(0), {d_concat_1}, {d_activation_1}, tpb__g5);
  Guard* g5 = gs->newGuard("relu_layer" + std::to_string(0), {v_concat_1}, {}, tp__g5, {});
  guard_log.push_back(g5);
  ////#pragma valve v_activation_1 = v1.init(call_num_activation_1, 1);
  auto v_activation_1 = v1.init(call_num_activation_1, 1);


  ////#pragma call_num call_num_max_pooling2d_1;
  auto call_num_max_pooling2d_1 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g6,{v_activation_1},{},{d_activation_1},{d_max_pooling2d_1}>>>CNN_squeezenetFluid_E::maxpool_layer_E(activation_1, max_pooling2d_1, 2, 2, call_num_max_pooling2d_1->p);
  auto tpb__g6 = std::bind(&CNN_squeezenetFluid_E::maxpool_layer_E, this, activation_1, max_pooling2d_1, 2, 2, call_num_max_pooling2d_1->p);
  auto tp__g6 = tf->newTask<decltype(tpb__g6)>("maxpool_layer" + std::to_string(0), {d_activation_1}, {d_max_pooling2d_1}, tpb__g6);
  Guard* g6 = gs->newGuard("maxpool_layer" + std::to_string(0), {v_activation_1}, {}, tp__g6, {});
  guard_log.push_back(g6);
  ////#pragma valve v_max_pooling2d_1 = v1.init(call_num_max_pooling2d_1, 1);
  auto v_max_pooling2d_1 = v1.init(call_num_max_pooling2d_1, 1);


  std::cout << "====================================================" << std::endl;


  ////#pragma call_num call_num_conv2d_2_s1;
  auto call_num_conv2d_2_s1 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g7,{v_max_pooling2d_1},{},{d_max_pooling2d_1},{d_conv2d_2_s1}>>>
  ////CNN_squeezenetFluid_E::convolution_layer_E(max_pooling2d_1, conv2d_2_s1, conv_weights3, 1.0 / rate, call_num_conv2d_2_s1->p);
  auto tpb__g7 = std::bind(&CNN_squeezenetFluid_E::convolution_layer_E, this, max_pooling2d_1, conv2d_2_s1, conv_weights3, 1.0 / rate, call_num_conv2d_2_s1->p);
  auto tp__g7 = tf->newTask<decltype(tpb__g7)>("convolution_layer" + std::to_string(2), {d_max_pooling2d_1}, {d_conv2d_2_s1}, tpb__g7);
  Guard* g7 = gs->newGuard("convolution_layer" + std::to_string(2), {v_max_pooling2d_1}, {}, tp__g7, {});
  guard_log.push_back(g7);
  ////#pragma valve v_conv2d_2_s1 = v1.init(call_num_conv2d_2_s1, conv2d_2_s1->c*rate);
  auto v_conv2d_2_s1 = v1.init(call_num_conv2d_2_s1, conv2d_2_s1->c*rate);


  ////#pragma call_num call_num_activation_2;
  auto call_num_activation_2 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g8,{v_conv2d_2_s1},{},{d_conv2d_2_s1},{d_activation_2, d_activation_22}>>>CNN_squeezenetFluid_E::relu_layer_E(conv2d_2_s1, activation_2, call_num_activation_2->p);
  auto tpb__g8 = std::bind(&CNN_squeezenetFluid_E::relu_layer_E, this, conv2d_2_s1, activation_2, call_num_activation_2->p);
  auto tp__g8 = tf->newTask<decltype(tpb__g8)>("relu_layer" + std::to_string(1), {d_conv2d_2_s1}, {d_activation_2, d_activation_22}, tpb__g8);
  Guard* g8 = gs->newGuard("relu_layer" + std::to_string(1), {v_conv2d_2_s1}, {}, tp__g8, {});
  guard_log.push_back(g8);
  ////#pragma valve v_activation_2 = v1.init(call_num_activation_2, 1);
  auto v_activation_2 = v1.init(call_num_activation_2, 1);


  ////#pragma call_num call_num_conv2d_2_e1;
  auto call_num_conv2d_2_e1 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g9,{v_activation_2},{},{d_activation_2},{d_conv2d_2_e1}>>>
  ////CNN_squeezenetFluid_E::convolution_layer_E(activation_2, conv2d_2_e1, conv_weights4, 1.0 / rate, call_num_conv2d_2_e1->p);
  auto tpb__g9 = std::bind(&CNN_squeezenetFluid_E::convolution_layer_E, this, activation_2, conv2d_2_e1, conv_weights4, 1.0 / rate, call_num_conv2d_2_e1->p);
  auto tp__g9 = tf->newTask<decltype(tpb__g9)>("convolution_layer" + std::to_string(3), {d_activation_2}, {d_conv2d_2_e1}, tpb__g9);
  Guard* g9 = gs->newGuard("convolution_layer" + std::to_string(3), {v_activation_2}, {}, tp__g9, {});
  guard_log.push_back(g9);
  ////#pragma valve v_conv2d_2_e1 = v1.init(call_num_conv2d_2_e1, conv2d_2_e1->c*rate);
  auto v_conv2d_2_e1 = v1.init(call_num_conv2d_2_e1, conv2d_2_e1->c*rate);


  ////#pragma call_num call_num_conv2d_2_e3;
  auto call_num_conv2d_2_e3 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g10,{v_activation_2},{},{d_activation_22},{d_conv2d_2_e3}>>>
  ////CNN_squeezenetFluid_E::convolution_layer_E(activation_2, conv2d_2_e3, conv_weights5, 1.0 / rate, call_num_conv2d_2_e3->p);
  auto tpb__g10 = std::bind(&CNN_squeezenetFluid_E::convolution_layer_E, this, activation_2, conv2d_2_e3, conv_weights5, 1.0 / rate, call_num_conv2d_2_e3->p);
  auto tp__g10 = tf->newTask<decltype(tpb__g10)>("convolution_layer" + std::to_string(4), {d_activation_22}, {d_conv2d_2_e3}, tpb__g10);
  Guard* g10 = gs->newGuard("convolution_layer" + std::to_string(4), {v_activation_2}, {}, tp__g10, {});
  guard_log.push_back(g10);
  ////#pragma valve v_conv2d_2_e3 = v1.init(call_num_conv2d_2_e3, conv2d_2_e3->c*rate);
  auto v_conv2d_2_e3 = v1.init(call_num_conv2d_2_e3, conv2d_2_e3->c*rate);


  ////#pragma call_num call_num_crop_2;
  auto call_num_crop_2 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g11,{v_conv2d_2_e1},{},{d_conv2d_2_e1},{d_crop_2}>>>CNN_squeezenetFluid_E::crop_layer_E(conv2d_2_e1, crop_2, 1, call_num_crop_2->p);
  auto tpb__g11 = std::bind(&CNN_squeezenetFluid_E::crop_layer_E, this, conv2d_2_e1, crop_2, 1, call_num_crop_2->p);
  auto tp__g11 = tf->newTask<decltype(tpb__g11)>("crop_layer" + std::to_string(1), {d_conv2d_2_e1}, {d_crop_2}, tpb__g11);
  Guard* g11 = gs->newGuard("crop_layer" + std::to_string(1), {v_conv2d_2_e1}, {}, tp__g11, {});
  guard_log.push_back(g11);
  ////#pragma valve v_crop_2 = v1.init(call_num_crop_2, 1);
  auto v_crop_2 = v1.init(call_num_crop_2, 1);


  ////#pragma call_num call_num_concat_2;
  auto call_num_concat_2 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g12,{v_crop_2},{},{d_crop_2, d_conv2d_2_e3},{d_concat_2}>>>CNN_squeezenetFluid_E::concat_layer_E(crop_2, conv2d_2_e3, concat_2, call_num_concat_2->p);
  auto tpb__g12 = std::bind(&CNN_squeezenetFluid_E::concat_layer_E, this, crop_2, conv2d_2_e3, concat_2, call_num_concat_2->p);
  auto tp__g12 = tf->newTask<decltype(tpb__g12)>("concat_layer" + std::to_string(1), {d_crop_2, d_conv2d_2_e3}, {d_concat_2}, tpb__g12);
  Guard* g12 = gs->newGuard("concat_layer" + std::to_string(1), {v_crop_2, v_conv2d_2_e3}, {}, tp__g12, {});
  guard_log.push_back(g12);
  ////#pragma valve v_concat_2 = v1.init(call_num_concat_2, 1);
  auto v_concat_2 = v1.init(call_num_concat_2, 1);


  ////#pragma call_num call_num_activation_3;
  auto call_num_activation_3 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g13,{v_concat_2},{},{d_concat_2},{d_activation_3}>>>CNN_squeezenetFluid_E::relu_layer_E(concat_2, activation_3, call_num_activation_3->p);
  auto tpb__g13 = std::bind(&CNN_squeezenetFluid_E::relu_layer_E, this, concat_2, activation_3, call_num_activation_3->p);
  auto tp__g13 = tf->newTask<decltype(tpb__g13)>("relu_layer" + std::to_string(2), {d_concat_2}, {d_activation_3}, tpb__g13);
  Guard* g13 = gs->newGuard("relu_layer" + std::to_string(2), {v_concat_2}, {}, tp__g13, {});
  guard_log.push_back(g13);
  ////#pragma valve v_activation_3 = v1.init(call_num_activation_3, 1);
  auto v_activation_3 = v1.init(call_num_activation_3, 1);


  ////#pragma call_num call_num_max_pooling2d_2;
  auto call_num_max_pooling2d_2 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g14,{v_activation_3},{},{d_activation_3},{d_max_pooling2d_2}>>>CNN_squeezenetFluid_E::maxpool_layer_E(activation_3, max_pooling2d_2, 2, 2, call_num_max_pooling2d_2->p);
  auto tpb__g14 = std::bind(&CNN_squeezenetFluid_E::maxpool_layer_E, this, activation_3, max_pooling2d_2, 2, 2, call_num_max_pooling2d_2->p);
  auto tp__g14 = tf->newTask<decltype(tpb__g14)>("maxpool_layer" + std::to_string(0), {d_activation_3}, {d_max_pooling2d_2}, tpb__g14);
  Guard* g14 = gs->newGuard("maxpool_layer" + std::to_string(0), {v_activation_3}, {}, tp__g14, {});
  guard_log.push_back(g14);
  ////#pragma valve v_max_pooling2d_2 = v1.init(call_num_max_pooling2d_2, 1);
  auto v_max_pooling2d_2 = v1.init(call_num_max_pooling2d_2, 1);


  std::cout << "====================================================" << std::endl;


  ////#pragma call_num call_num_flatten_;
  auto call_num_flatten_ = new __fluid__<int>(new int(0));
  ////#pragma task <<<g15,{v_max_pooling2d_2},{},{d_max_pooling2d_2},{d_flatten_}>>>CNN_squeezenetFluid_E::flatten_E(max_pooling2d_2, flatten_, call_num_flatten_->p);
  auto tpb__g15 = std::bind(&CNN_squeezenetFluid_E::flatten_E, this, max_pooling2d_2, flatten_, call_num_flatten_->p);
  auto tp__g15 = tf->newTask<decltype(tpb__g15)>("flatten" + std::to_string(0), {d_max_pooling2d_2}, {d_flatten_}, tpb__g15);
  Guard* g15 = gs->newGuard("flatten" + std::to_string(0), {v_max_pooling2d_2}, {}, tp__g15, {});
  guard_log.push_back(g15);
  ////#pragma valve v_flatten_ = v1.init(call_num_flatten_, 1);
  auto v_flatten_ = v1.init(call_num_flatten_, 1);


  ////#pragma call_num call_num_dense_1;
  auto call_num_dense_1 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g16,{v_flatten_},{},{d_flatten_},{d_dense_1}>>>CNN_squeezenetFluid_E::dense_layer_E(flatten_, dense_1, dense_weights1, 1.0 / rate, call_num_dense_1->p);
  auto tpb__g16 = std::bind(&CNN_squeezenetFluid_E::dense_layer_E, this, flatten_, dense_1, dense_weights1, 1.0 / rate, call_num_dense_1->p);
  auto tp__g16 = tf->newTask<decltype(tpb__g16)>("dense_layer" + std::to_string(0), {d_flatten_}, {d_dense_1}, tpb__g16);
  Guard* g16 = gs->newGuard("dense_layer" + std::to_string(0), {v_flatten_}, {}, tp__g16, {});
  guard_log.push_back(g16);
  ////#pragma valve v_dense_1 = v1.init(call_num_dense_1, dense_1->c*rate);
  auto v_dense_1 = v1.init(call_num_dense_1, dense_1->c*1);


  ////#pragma call_num call_num_activation_4;
  auto call_num_activation_4 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g17,{v_dense_1},{},{d_dense_1},{d_activation_4}>>>CNN_squeezenetFluid_E::relu_layer_E(dense_1, activation_4, call_num_activation_4->p);
  auto tpb__g17 = std::bind(&CNN_squeezenetFluid_E::relu_layer_E, this, dense_1, activation_4, call_num_activation_4->p);
  auto tp__g17 = tf->newTask<decltype(tpb__g17)>("relu_layer" + std::to_string(3), {d_dense_1}, {d_activation_4}, tpb__g17);
  Guard* g17 = gs->newGuard("relu_layer" + std::to_string(3), {v_dense_1}, {}, tp__g17, {});
  guard_log.push_back(g17);
  ////#pragma valve v_activation_4 = v1.init(call_num_activation_4, 1);
  auto v_activation_4 = v1.init(call_num_activation_4, 1);


  std::cout << "====================================================" << std::endl;


  ////#pragma call_num call_num_dense_2;
  auto call_num_dense_2 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g18,{v_activation_4},{},{d_activation_4},{d_dense_2}>>>CNN_squeezenetFluid_E::dense_layer_E(activation_4, dense_2, dense_weights2, 1.0 / rate, call_num_dense_2->p);
  auto tpb__g18 = std::bind(&CNN_squeezenetFluid_E::dense_layer_E, this, activation_4, dense_2, dense_weights2, 1.0 , call_num_dense_2->p);
  auto tp__g18 = tf->newTask<decltype(tpb__g18)>("dense_layer" + std::to_string(1), {d_activation_4}, {d_dense_2}, tpb__g18);
  Guard* g18 = gs->newGuard("dense_layer" + std::to_string(1), {v_activation_4}, {}, tp__g18, {});
  guard_log.push_back(g18);
  ////#pragma valve v_dense_2 = v1.init(call_num_dense_2, dense_2->c*rate);
  auto v_dense_2 = v1.init(call_num_dense_2, dense_2->c*1);


  ////#pragma call_num call_num_activation_5;
  auto call_num_activation_5 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g19,{v_dense_2},{},{d_dense_2},{d_activation_5}>>>CNN_squeezenetFluid_E::relu_layer_E(dense_2, activation_5, call_num_activation_5->p);
  auto tpb__g19 = std::bind(&CNN_squeezenetFluid_E::relu_layer_E, this, dense_2, activation_5, call_num_activation_5->p);
  auto tp__g19 = tf->newTask<decltype(tpb__g19)>("relu_layer" + std::to_string(4), {d_dense_2}, {d_activation_5}, tpb__g19);
  Guard* g19 = gs->newGuard("relu_layer" + std::to_string(4), {v_dense_2}, {}, tp__g19, {});
  guard_log.push_back(g19);
  ////#pragma valve v_activation_5 = v1.init(call_num_activation_5, 1);
  auto v_activation_5 = v1.init(call_num_activation_5, 1);


  ////#pragma call_num call_num_dense_3;
  auto call_num_dense_3 = new __fluid__<int>(new int(0));
  ////#pragma task <<<g20,{v_activation_5},{},{d_activation_5},{d_dense_3}>>>CNN_squeezenetFluid_E::dense_layer_E(activation_5, dense_3, dense_weights3, 1.0 / rate, call_num_dense_3->p);
  auto tpb__g20 = std::bind(&CNN_squeezenetFluid_E::dense_layer_E, this, activation_5, dense_3, dense_weights3, 1.0 , call_num_dense_3->p);
  auto tp__g20 = tf->newTask<decltype(tpb__g20)>("dense_layer" + std::to_string(2), {d_activation_5}, {d_dense_3}, tpb__g20);
  Guard* g20 = gs->newGuard("dense_layer" + std::to_string(2), {v_activation_5}, {}, tp__g20, {});
  guard_log.push_back(g20);
  ////#pragma valve v_dense_3 = v1.init(call_num_dense_3, dense_3->c*rate);
  auto v_dense_3 = v1.init(call_num_dense_3, dense_3->c*1);


  ////#pragma call_num call_num_activation_6;
  auto call_num_activation_6 = new __fluid__<int>(new int(0));
  ////#pragma valve v_activation_6 = v1.init(call_num_activation_6, 1);
  auto v_activation_6 = v1.init(call_num_activation_6, 1);
  ////#pragma task <<<g21,{v_dense_3},{},{d_dense_3},{d_activation_6}>>>CNN_squeezenetFluid_E::softmax_layer_E(dense_3, activation_6, call_num_activation_6->p);
  auto tpb__g21 = std::bind(&CNN_squeezenetFluid_E::softmax_layer_E, this, dense_3, activation_6, call_num_activation_6->p);
  auto tp__g21 = tf->newTask<decltype(tpb__g21)>("softmax_layer" + std::to_string(0), {d_dense_3}, {d_activation_6}, tpb__g21);
  Guard* g21 = gs->newGuard("softmax_layer" + std::to_string(0), {v_dense_3}, {v_activation_6}, tp__g21, {});
  guard_log.push_back(g21);
  /////g21->set_leaf();
  g21->set_leaf();

  gs->synctask(tp__g21);

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


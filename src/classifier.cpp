/*
@author: liangh
@contact: huangliang198911@yahoo.com
*/
#include <iostream>
#include <cstring>
#include <gflags/gflags.h>
#include "convert.h"
#include "classifier.h"
#include "math_util.h"
#include "file_util.h"

namespace test {
namespace tensorrt {

Classifier::Classifier(
    const string& engine_file,
    const string& output_blob_file,
    const string& mean_file,
    const string& label_file,
    float stdvar,
    int base_size,
    int top_k,
    bool is_bgr) :
    stdvar_(stdvar), base_size_(base_size), is_bgr_(is_bgr) {
  // read engine file
  // To generate an engine file, you can use a tool in app/trt_tool_gen
  std::ifstream in_file(engine_file.c_str(), std::ios::in | std::ios::binary);
  if (!in_file.is_open()) {
    std::cout << "Failed to open engine file" << std::endl;
    exit(1);
  }
  std::streampos begin, end;
  begin = in_file.tellg();
  in_file.seekg(0, std::ios::end);
  end = in_file.tellg();
  std::size_t size = end - begin;
  in_file.seekg(0, std::ios::beg);
  std::unique_ptr<unsigned char[]> engine_data(new unsigned char[size]);
  in_file.read((char*)engine_data.get(), size);
  in_file.close();
  // create a engine
  runtime_ = nvinfer1::createInferRuntime(gLogger);
  engine_ = runtime_->deserializeCudaEngine(
     (const void*)engine_data.get(), size, nullptr);
  if (engine_ == nullptr) {
    std::cout << "Fail in rebuilding engine..." << std::endl;
  }
  context_ = engine_->createExecutionContext();
  if (context_ == nullptr) {
    std::cout << "Fail in creating execution context..." << std::endl;
  }
  // get engine's max batch size
  n_ = engine_->getMaxBatchSize();
  std::cout << "setting gpu memory !!!" << std::endl;
  // get input index on buffers_
  input_index_ = engine_->getBindingIndex("data");
  // get output_blob_names, then get output index on buffers_
  auto output_blob_names = ReadFileByLine(output_blob_file);
  output_index_ = engine_->getBindingIndex(output_blob_names[0].c_str());
  std::cout << "setting gpu memory !!!" << std::endl;
  // get input& output size
  Dims3 input_dims = static_cast<Dims3&&>(engine_->getBindingDimensions(input_index_));
  Dims3 output_dims = static_cast<Dims3&&>(engine_->getBindingDimensions(output_index_));
  std::cout << "setting gpu memory !!!" << std::endl;
  c_ = input_dims.d[0];
  h_ = input_dims.d[1];
  w_ = input_dims.d[2];
  output_c_ = output_dims.d[0];
  output_h_ = output_dims.d[1];
  output_w_ = output_dims.d[2];
  input_geometry_ = cv::Size(w_, h_);
  std::cout << "setting gpu memory !!!" << std::endl;
  // caculate the volume of (image, channel & output)
  vol_chl_ = h_ * w_;
  int vol_img = c_ * vol_chl_;
  vol_output_ = output_c_ * output_h_ * output_w_;
  // allocate gpu memory for buffers_
  std::cout << "setting gpu memory !!!" << std::endl;
  input_size_ = vol_img * n_ * sizeof(float);
  output_size_ = output_c_ * output_h_ * output_w_ * n_ * sizeof(float);
  CHECK(cudaMalloc(&buffers_[input_index_], input_size_));
  CHECK(cudaMalloc(&buffers_[output_index_], output_size_));
  // allocate cpu memory for input_batch_ & output_batch_;
  std::cout << "gpu memory created !!!" << std::endl;
  input_batch_ = new float[vol_img * n_];
  output_batch_ = new float[vol_output_ * n_];
  // initialize cuda stream
  CHECK(cudaStreamCreate(&stream_));
  // load labels
  labels_ = ReadFileByLine(label_file);
  // set mean
  SetMean(mean_file);
  std::cout << "mean set !!!" << std::endl;
  top_k_ = (labels_.size() < top_k) ? labels_.size() : top_k;
}

void Classifier::SetMean(const string& mean_file) {
  // create RGB mean mat from mean list file 
  vector<string> mean_vec = ReadFileByLine(mean_file);
  for (int i = 0; i < mean_vec.size(); i++) {
    if (is_bgr_) {
      bgr_mean_[i] = atof(mean_vec[i].c_str());
    } else {
      rgb_mean_[i] = atof(mean_vec[i].c_str());
    }
  }
  vector<cv::Mat> mean_channels;
  // if the channel order is bgr
  if (is_bgr_) {
    for (int i=0; i < 3; i++) {
      cv::Mat channel(h_, w_, CV_32FC1, cv::Scalar(bgr_mean_[i]));
      mean_channels.push_back(channel);
    }
  // if the channel order is rgb
  } else {
    for (int i=0; i < 3; i++) {
      cv::Mat channel(h_, w_, CV_32FC1, cv::Scalar(rgb_mean_[i]));
      mean_channels.push_back(channel);
    }
  }
  cv::merge(mean_channels, mean_);
}

void Classifier::WrapInputLayer(vector<vector<cv::Mat>>* input_batch) {
  // allocate input_batch memory
  float* input_data = input_batch_;
  for (int j = 0; j < n_; j++) {
    vector<cv::Mat> input_channels;
    // allocate one channel's memory pointer
    for (int i = 0; i < c_; ++i) {
      cv::Mat channel(h_, w_, CV_32FC1, input_data);
      input_channels.push_back(channel);
      input_data += vol_chl_;
    }
    // collect all pointers to input_batch
    input_batch->push_back(input_channels);
  }
}

void Classifier::Preprocess(const vector<cv::Mat>& imgs, vector<vector<cv::Mat>>* input_batch) {
  for (int i = 0; i < n_; i++) {
    // get image, if not exists, then use an all-zero mat instead
    cv::Mat sample;
    if (i < imgs.size()) {
      sample = imgs[i];
    } else {
      if (base_size_ != 0) {
        sample = cv::Mat(cv::Size(base_size_, base_size_), CV_8UC3, cv::Scalar(0, 0, 0));
      } else {
        sample = cv::Mat(input_geometry_, CV_8UC3, cv::Scalar(0, 0, 0));
      }
    }
    // change channel order
    if (!is_bgr_) {
      cv::cvtColor(sample, sample, cv::COLOR_BGR2RGB);
    }
    cv::Mat sample_resized;
    // if base_size is not 0, resize first then crop 
    if (base_size_ != 0) {
      float short_edge = float ((sample.cols < sample.rows) ? sample.cols : sample.rows);
      float resize_ratio = float(base_size_) / short_edge;
      cv::Size resize_size =  cv::Size(
          int(resize_ratio * sample.cols), int(resize_ratio * sample.rows));
      cv::resize(sample, sample_resized, resize_size);
      auto xx = int((sample_resized.cols - w_) / 2);
      auto yy = int((sample_resized.rows - h_) / 2);
      auto crop_rect = cv::Rect(xx, yy, w_, h_);
      sample_resized = sample_resized(crop_rect);
    } else {
      // if base size is 0, resize the input
      cv::resize(sample, sample_resized, input_geometry_);
    }
    cv::Mat sample_float;
    sample_resized.convertTo(sample_float, CV_32FC3);
    // normalize image, first subtracted by mean mat, then divided by standard variance
    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);
    sample_normalized /= stdvar_;
    // get input mat pointer
    vector<cv::Mat>* input_channels = &(input_batch->at(i));
    // split input into channels to fit-in mat pointer
    cv::split(sample_normalized, *input_channels);
  }
}

vector<vector<float>> Classifier::Predict(const vector<cv::Mat>& imgs) {
  // declaration of input batch vector
  vector<vector<cv::Mat>> input_batch;
  // call the function to allocate image pointers to net input pointer
  WrapInputLayer(&input_batch);
  // do preprocess
  Preprocess(imgs, &input_batch);
  // copy input data from host to gpu device
  CHECK(cudaMemcpyAsync(
      buffers_[input_index_], input_batch_, input_size_, cudaMemcpyHostToDevice, stream_));
  // do inference on gpu
  context_->enqueue(n_, buffers_, stream_, nullptr);
  // copy output data from gpu device to host
  CHECK(cudaMemcpyAsync(
      output_batch_, buffers_[output_index_], output_size_, cudaMemcpyDeviceToHost, stream_));
  // do sync
  cudaStreamSynchronize(stream_);
  // copy output data to vector
  vector<vector<float>> outputs;
  for (int i = 0; i < imgs.size(); ++i) {
    const float* local_begin = output_batch_ + i * vol_output_;
    const float* local_end = local_begin + vol_output_;
    outputs.push_back(vector<float>(local_begin, local_end));
  }
  return outputs;
}

vector<vector<Prediction>> Classifier::Classify(const vector<cv::Mat>& imgs) {
  // get network output
  auto outputs = Predict(imgs);
  vector<vector<Prediction>> results;
  for (auto& output:outputs) {
    // calculate top k output
    vector<int> maxN = Argmax(output, top_k_);
    vector<Prediction> result;
    // push top k output into result
    for (int i = 0; i < top_k_; ++i) {
      int idx = maxN[i];
      result.emplace_back(std::make_pair(labels_[idx], output[idx]));
    }
    results.push_back(result);
  }
  return results;
}

}  // namespace tensorrt
}  // namespace test

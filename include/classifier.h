/*
@author: liangh
@contact: huangliang198911@yahoo.com
*/
#ifndef INCLUDE_CLASSIFIER_H_
#define INCLUDE_CLASSIFIER_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <cuda_runtime_api.h>
#include "common.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

using std::vector;
using std::string;

namespace test {
namespace tensorrt {

// Pair (label, confidence) representing a prediction
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  /* Construct function
  param @engine_file	the TensorRT engine file
	@output_blob_file	indicate the output layer name, which is saved as a list file
	@mean_file	the TensorRT model mean file, which is different from caffe mean file.
			Because TensorRT lib cannot read binaryproto,
			we use a list file to indicate the mean value of each input channel
	@label_file	model's label list file
	@stdvar		stdvar is the standard variance of input data,
			some model's are trained with data normalized to N(0,1)
	@base_size	is the resize size of input image, if the preprocess include resize&crop.
			if only need resize, then base_size should be set to 0
	@top_k		for multi-classification softmax output, we want to get top_k output
	@is_bgr		to indicate whether the model is trained with (b, g, r) ordered image
  */
  Classifier(
      const string& engine_file, 
      const string& output_blob_file,
      const string& mean_file,
      const string& label_file,
      float stdvar = 1.0,
      int base_size = 0,
      int top_k = 1,
      bool is_bgr = true);
  // Destruction function, release memory allocated on cpu&gpu, release tensorrt handlers
  ~Classifier() {
    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
    cudaStreamDestroy(stream_);
    CHECK(cudaFree(buffers_[0]));
    CHECK(cudaFree(buffers_[1]));
    delete[] input_batch_;
    delete[] output_batch_;
  }
  // Classify batched images
  vector<vector<Prediction>> Classify(const vector<cv::Mat>& imgs);
  // Get batch size of model
  int GetBatchSize() {
    return n_;
  }
 private:
  // Set mean mat with mean_file
  void SetMean(const string& mean_file);
  // Allocate input layer pointer to input image mat
  void WrapInputLayer(vector<vector<cv::Mat>>* input_batch);
  // Preprocess the images to fit-in input layer
  void Preprocess(const vector<cv::Mat>& imgs, vector<vector<cv::Mat>>* input_batch);
  // Get network output
  vector<vector<float>> Predict(const vector<cv::Mat>& imgs);
 private:
  // tensorrt handlers
  nvinfer1::IRuntime* runtime_;
  nvinfer1::ICudaEngine* engine_;
  nvinfer1::IExecutionContext* context_;
  cudaStream_t stream_;
  // input shape
  int n_, c_, h_, w_;
  // input image shape
  cv::Size input_geometry_;
  // resize image base size
  int base_size_;
  // output shape
  int output_c_, output_h_, output_w_;
  // input size of (one channel, one output volume)
  int vol_chl_, vol_output_;
  // input&output volume size
  size_t input_size_, output_size_;
  // input&output gpu memory pointers
  void* buffers_[2];
  // input&output cpu memory pointers
  float* input_batch_;
  float* output_batch_;
  // input&output idx on buffers_
  int input_index_, output_index_;
  // default image mean value 
  float rgb_mean_[3] = {122.7717, 115.9465, 102.9801};
  float bgr_mean_[3] = {102.9801, 115.9465, 122.7717};
  // mean mat
  cv::Mat mean_;
  // model channel order
  bool is_bgr_;
  // input image standard variance
  float stdvar_;
  // top k output
  int top_k_;
  // class labels list
  vector<std::string> labels_;
};
}  // namespace tensorrt
}  // namespace test
#endif //INCLUDE_CLASSIFIER_H_

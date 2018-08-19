/*
@author: liangh
@contact: huangliang198911@yahoo.com
*/
#ifndef INCLUDE_CONVERT_H_ 
#define INCLUDE_CONVERT_H_ 

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <cstring>
#include <algorithm>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "common.h"

static Logger gLogger;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

namespace test {
namespace tensorrt {

/*The function to transfer a caffe model to TensorRT engine
param	@model_file		caffe_prototxt file path
	@pretraied_file		caffemodel file path
	@outputs		network outputs
	@batch_size		batch size which we want to run with
	@plugin_factory		factory for plugin layers
	@trt_model_stream	output stream for TensorRT model
*/
inline bool CaffeToTRTModel(
    const std::string& model_file,
    const std::string& pretrained_file,
    const std::vector<std::string>& outputs,
    unsigned int batch_size,
    nvcaffeparser1::IPluginFactory* plugin_factory,
    nvinfer1::IHostMemory*& trt_model_stream) {
  // create engine builder
  nvinfer1::IBuilder* builder = createInferBuilder(gLogger);
  // declare the caffe model parser and network handler
  nvinfer1::INetworkDefinition* network = builder->createNetwork();
  nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
  // set plugin factory if there exists
  if (plugin_factory != nullptr) {
    parser->setPluginFactory(plugin_factory);
  }
  std::cout << "Begin parsing model..." << std::endl;
  // parse network from caffe prototxt& caffemodel
  const nvcaffeparser1::IBlobNameToTensor* blob_name_to_tensor = parser->parse(
      model_file.c_str(),
      pretrained_file.c_str(),
      *network,
      DataType::kFLOAT);
  std::cout << "End parsing model..." << std::endl;
  // specify which tensors are outputs
  for (auto& s : outputs) {
    network->markOutput(*blob_name_to_tensor->find(s.c_str()));
  }
  // set the engine builder parameters
  builder->setMaxBatchSize(batch_size);
  builder->setMaxWorkspaceSize(16 << 20);
  // build engine
  std::cout << "Begin building engine..." << std::endl;
  nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
  if (engine == nullptr) {
    std::cout << "Fail in building engine..." << std::endl;
    return false;
  }
  std::cout << "End building engine..." << std::endl;
  // release network & parser
  network->destroy();
  parser->destroy();
  // serialize the engine, then close everything down
  trt_model_stream = engine->serialize();
  engine->destroy();
  builder->destroy();
  nvcaffeparser1::shutdownProtobufLibrary();
  return true;
}

inline bool SaveTRTModel(
    const std::string& engine_file, 
    nvinfer1::IHostMemory* trt_model_stream) {
  // create output file handler
  std::ofstream outfile(engine_file.c_str(), std::ios::out | std::ios::binary);
  if (!outfile.is_open()) {
    std::cout << "Failed to open engine file" << std::endl;
    return false;
  }
  // get output data pointer
  unsigned char* p = (unsigned char*)(trt_model_stream->data());
  // write output to output file
  outfile.write((char*)p, trt_model_stream->size());
  outfile.close();
  return true;
}

} // namespace tensorrt
} // namespace test
#endif // INCLUDE_CONVERT_H_ 

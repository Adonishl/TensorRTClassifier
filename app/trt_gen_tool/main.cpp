#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <gflags/gflags.h>
#include "convert.h"
#include "file_util.h"

using test::tensorrt::ReadFileByLine;
using test::tensorrt::CaffeToTRTModel;
using test::tensorrt::SaveTRTModel;

DEFINE_string(model_file, "", "caffe model file path");
DEFINE_string(pretrained_file, "", "caffe model pretrained weights file path");
DEFINE_string(output_blob_file, "", "model output blob names path");
DEFINE_int32(batch_size, 8, "batch size of output engine");
DEFINE_string(output_path, "", "transfered TensorRT engine output path");


int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  auto output_names = ReadFileByLine(FLAGS_output_blob_file); 
  nvinfer1::IHostMemory *trt_model_stream{ nullptr };
  if (CaffeToTRTModel(
      FLAGS_model_file,
      FLAGS_pretrained_file,
      output_names,
      FLAGS_batch_size,
      nullptr,
      trt_model_stream)) {
    if (SaveTRTModel(
        FLAGS_output_path,
        trt_model_stream )) {
      trt_model_stream->destroy();
      std::cout << "Succeed in converting to a TensorRT engine" << std::endl;
    } else {
      trt_model_stream->destroy();
      return -1;
    }
  } else {
    trt_model_stream->destroy();
    return -1;
  }

  return 0;
}

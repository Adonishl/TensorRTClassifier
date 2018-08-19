#include <string>
#include <vector>
#include <memory>
#include <cassert>
#include <algorithm>
#include <gflags/gflags.h>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include "file_util.h"
#include "time_util.h"
#include "classifier.h"

using namespace std;
using namespace boost;
using test::tensorrt::ReadFileByLine;
using test::tensorrt::WriteFileByLine;
using test::tensorrt::GetTimeMilli;
using test::tensorrt::Classifier;


DEFINE_string(dataset_dir, "", "test dataset directory");
DEFINE_string(gt_text, "", "ground truth text file path");
DEFINE_string(output_path, "", "output file path");
DEFINE_bool(verbose, true, "whether to print some info");
DEFINE_string(engine_file, "", "trt engine's path");
DEFINE_string(output_blob_file, "", "model output blob names path");
DEFINE_string(mean_file, "", "mean file's path");
DEFINE_string(label_file, "", "class labels text file");
DEFINE_double(stdvar, 1.0, "input standard variance");
DEFINE_int32(base_size, 0, "resize base size before cropped");
DEFINE_int32(top_k, 1, "output top k results");

// Join the dirctory with file name
// @Input	file_dir & file_name
// @Return	file_path
string JoinPath(const string& file_dir, const string& file_name) {
  filesystem::path fs_file_dir = filesystem::path(file_dir);
  filesystem::path fs_file_name = filesystem::path(file_name);
  fs_file_dir /= fs_file_name;
  return fs_file_dir.string();
}

// Split the ground truth label with the image name
// @Input	pair_list
// @Return	image_list & gt_label_list
void SplitImageWithLabel(
    const string& dataset_dir,
    const vector<string>& pair_list,
    vector<string>& image_list,
    vector<string>& gt_label_list) {
  for(auto& pair : pair_list) {
    vector<string> fields;
    split( fields, pair, is_any_of(" "));
    image_list.emplace_back(JoinPath(dataset_dir, fields[0]));
    gt_label_list.emplace_back(fields[1]);
  }
}

vector<vector<string>> SplitImageList(
    const vector<string>& image_list,
    int batch_size) {
  int enlist_num = 0;
  int total_num = image_list.size();
  vector<vector<string>> image_slices;
  while (enlist_num != total_num) {
    if ((total_num - enlist_num) < batch_size) {
      image_slices.push_back(vector<string>(image_list.begin() + enlist_num, image_list.end()));
      enlist_num = total_num;
      break;
    }
    auto local_begin = image_list.begin() + enlist_num;
    image_slices.push_back(vector<string>(local_begin, local_begin + batch_size));
    enlist_num += batch_size;
  }
  return image_slices;
}

int main(int argc, char** argv) {
  /******** Parse flags ********/
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  /******** Init classifier ********/
  unique_ptr<Classifier> classifier;
  classifier.reset(new Classifier(
      FLAGS_engine_file, FLAGS_output_blob_file,
      FLAGS_mean_file, FLAGS_label_file,
      FLAGS_stdvar, FLAGS_base_size, FLAGS_top_k));
  int batch_size = classifier->GetBatchSize();
  std::cout << "batch size: " << batch_size << std::endl;
  /******** Read ground truth file ********/
  vector<string> image_label_pair_list;
  image_label_pair_list = ReadFileByLine(FLAGS_gt_text);
  // test images' path list
  vector<string> image_list;
  // test images' ground truth label list
  vector<string> gt_label_list;
  SplitImageWithLabel(FLAGS_dataset_dir, image_label_pair_list, image_list, gt_label_list);
  // print image_list & gt_labels_list
  assert(image_list.size() == gt_label_list.size());
  int total_num = image_list.size();
  if (FLAGS_verbose) {
    for (int i = 0; i < total_num; i++) {
      cout <<"image path: " << image_list[i] << "\t" 
          << "ground truth label: " << gt_label_list[i] << endl;
    }
  }
  
  /******** Process images in a mini-batch ********/
  // split images into slices
  auto image_slices = SplitImageList(image_list, batch_size);
  // print splitted image slices
  if (FLAGS_verbose) {
    for (int i = 0; i < image_slices.size(); i++) {
      cout << "slice " << i << ":" << endl;
      for (int j = 0; j < image_slices[i].size(); j++) {
        cout << image_slices[i][j] << endl; 
      }
    }
  }
  vector<vector<string>> result_list;
  //process images in batch
  uint64_t infer_time = 0;
  for (int i = 0; i < image_slices.size(); i++) {
    cout << "processing " << i <<"th slice" << endl;
    // read images in batch
    vector<cv::Mat> mats;
    for (int j = 0; j < image_slices[i].size(); j++) {
      mats.push_back(cv::imread(image_slices[i][j]));
    }
    // do inference
    auto tic = GetTimeMilli();
    auto results = classifier->Classify(mats);
    auto toc = GetTimeMilli();
    infer_time += (toc - tic);
    for (int j = 0; j < results.size(); j++) {
      auto& result = results[j];
      vector<string> predicts;
      for (int k = 0; k < result.size(); k++) {
        predicts.emplace_back(result[k].first);
      }
      // append test result into result list
      result_list.emplace_back(predicts);
    }
  }
  assert(total_num == result_list.size());
  cout << "Infer Time Cost: " << infer_time << "ms" << endl;
  cout << "fps: " << float(total_num) * 1000.0 / infer_time << endl;
  /******** Compare the result & ground truth list ********/
  int correct_num = 0;
  vector<string> output_list;
  for (int i = 0; i < total_num; i++) {
    auto& result = result_list[i];
    if (find(result.begin(), result.end(), gt_label_list[i]) != result.end()) {
      correct_num++;
    }
    if (FLAGS_output_path != "") {
      string outputline = "Top " + std::to_string(FLAGS_top_k) + ":";
      for (int j = 0; j < result.size(); j++) {
        outputline += result[j];
        outputline += "\t";
      }
      outputline += ("GT: " + gt_label_list[i]);
      output_list.emplace_back(outputline);
    }
  }
  float accuracy = float(correct_num) / float(total_num);
  cout << "Total num: " << total_num << endl;
  cout << "Correct num: " << correct_num << endl;
  cout << "Accuracy: " << accuracy << endl;
  if (FLAGS_output_path != "") {
    WriteFileByLine(FLAGS_output_path, output_list);
  }
  return 0;
}

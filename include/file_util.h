/*
@author: liangh
@contact: huangliang198911@yahoo.com
*/
#ifndef INCLUDE_FILE_UTIL_H_
#define INCLUDE_FILE_UTIL_H_

#include <fstream>
#include <sstream>
#include <vector>
#include <string>

namespace test {
namespace tensorrt {

// read a file list
inline std::vector<std::string> ReadFileByLine(const std::string& in_path) {
  std::vector<std::string> output_lines;
  std::string line;
  std::ifstream infile(in_path.c_str());
  while (std::getline(infile, line)) {
    output_lines.push_back(line);
  }
  return output_lines;
}

// write a file list
inline void WriteFileByLine(
    const std::string& out_path, const std::vector<std::string>& lines) {
  std::ofstream outfile(out_path.c_str());
  for (int i = 0; i < lines.size(); i++) {
    outfile << lines[i] << "\n";
  }
  outfile.close();
}


}  // namespace tensorrt
}  // namespace test
#endif  // INCLUDE_FILE_UTIL_H_

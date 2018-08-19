/*
@author: liangh
@contact: huangliang198911@yahoo.com
*/
#ifndef INCLUDE_TIME_H_
#define INCLUDE_TIME_H_

#include <time.h>
#include <stdint.h>

namespace test {
namespace tensorrt {

// get current timestamp in nanotime
inline uint64_t GetTimeNano() {
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return (uint64_t)(t.tv_sec * 1000*1000*1000) + (uint64_t)(t.tv_nsec);
}

// get current timestamp in millitime
inline uint64_t GetTimeMilli() {
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return (uint64_t)(t.tv_sec * 1000) + (uint64_t)(t.tv_nsec/1000/1000);
}

}  // namespace tensorrt
}  // namespace test
#endif  // INCLUDE_TIME_H_


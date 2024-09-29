#ifndef UTILS_H
#define UTILS_H

#include <execinfo.h>

#include <cstdio>

#if defined(__GNUC__)
#define FORCE_INLINE __attribute__((always_inline)) inline
#else
#define FORCE_INLINE inline
#endif

[[noreturn]] inline void error(const char* message) {
  if (message == nullptr) {
    std::printf("ERROR\n");
  } else {
    std::printf("ERROR: %s\n", message);
  }

  std::printf("\nPrinting backtrace:\n");
  void* callstack_arr[256];
  int n_frames = backtrace(callstack_arr, 256);
  char** strings = backtrace_symbols(callstack_arr, n_frames);
  for (int i = 0; i < n_frames; ++i) {
    std::printf("%s\n", strings[i]);
  }
  std::free(strings);

  exit(1);
}

inline void require(bool condition, const char* message) {
  if (!condition) error(message);
}

#endif /* UTILS_H */

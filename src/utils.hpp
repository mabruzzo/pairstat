#ifndef UTILS_H
#define UTILS_H

#include <cstdio>

[[noreturn]] inline void error(const char* message){
  if (message == nullptr){
    printf("ERROR\n");
  } else {
    printf("ERROR: %s\n", message);
  }
  exit(1);
}

#endif /* UTILS_H */

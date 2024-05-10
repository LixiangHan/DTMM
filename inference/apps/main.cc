/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <stdio.h>
#include "main_functions.h"
#include "uart.h"

#define INFERENCE 1
// This is the default main used on systems that have the standard C entry
// point. Other devices (for example FreeRTOS or ESP32) that have different
// requirements for entry code (like an app_main function) should specialize
// this main.cc file in a target-specific subfolder.
int main(int argc, char* argv[]) {
  uint32_t avg_inference_time = 0;
  uart_init();
  setup();
  for (int i = 0; i < INFERENCE; i++) {
    avg_inference_time += loop() / INFERENCE;
  }
  printf("Inference %d times.\n", INFERENCE);
  printf("Average CPU Cycles = %ld.\n", avg_inference_time);
  printf("Finished.");
  return 0;
}

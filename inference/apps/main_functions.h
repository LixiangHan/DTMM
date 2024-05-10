#ifndef APPS_MAIN_FUNCTIONS_H_
#define APPS_MAIN_FUNCTIONS_H_

// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C" {
#endif

// Initializes all data needed for the example. The name is important, and needs
// to be setup() for Arduino compatibility.
void setup();

// Runs one iteration of data gathering and inference. This should be called
// repeatedly from the application code. The name needs to be loop() for Arduino
// compatibility.
uint32_t loop();

#ifdef __cplusplus
}
#endif

#endif  // RESNET_9_COMPRESSED_MAIN_FUNCTIONS_H_
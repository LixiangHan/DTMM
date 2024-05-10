#include <stdio.h>

#include "config.h"
#include "tflite_model.h"
#include "input_data.h"
#include "main_functions.h"
#include "output_handler.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <ARMCM55.h>
#include <pmu_armv8.h>

namespace
{
    tflite::ErrorReporter *error_reporter = nullptr;
    const tflite::Model *model = nullptr;
    tflite::MicroInterpreter *interpreter = nullptr;
    TfLiteTensor *input = nullptr;
    TfLiteTensor *output = nullptr;

    constexpr int kTensorArenaSize = ARENA_SIZE;
    uint8_t tensor_arena[kTensorArenaSize];
}

volatile uint32_t cycle_count_before = 0;
volatile uint32_t cycle_count_after = 0;

void setup()
{
    printf("RUNING: %s\n", APP_NAME);
    ARM_PMU_Enable();

    tflite::InitializeTarget();

    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(tflite_model);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        TF_LITE_REPORT_ERROR(error_reporter,
                             "Model provided is schema version %d not equal "
                             "to supported version %d.",
                             model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    static tflite::AllOpsResolver resolver;

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        printf("AllocateTensors() failed\n");
        return;
    }
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Copy an image into the memory area used for the input
    for (int i = 0; i < INPUT_CHANNEL * INPUT_HEIGHT * INPUT_WIDTH; i++)
    {
        input->data.int8[i] = input_data[i];
    }
}

uint32_t loop()
{
    TfLiteStatus status;
    static int count = 0;
    printf("Inference %d: \n", count);

    ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);

    cycle_count_before = ARM_PMU_Get_CCNTR();
    status = interpreter->Invoke();

    cycle_count_after = ARM_PMU_Get_CCNTR();

    ARM_PMU_CNTR_Disable(PMU_CNTENCLR_CCNTR_ENABLE_Msk);
    ARM_PMU_CYCCNT_Reset();

    if (kTfLiteOk != interpreter->Invoke())
    {
        // TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
        printf("Invoke failed.\n");
    }
    else
    {
        printf("    CPU Cycles = %ld\n", cycle_count_after - cycle_count_before);
    }
    output = interpreter->output(0);
    for (int i = 0; i < 10; i++) 
    {
        printf("%d ", output->data.int8[i]);
    }
    printf("\n");
    // for (int i = 0; i <16; i++) 
    // {
    //     printf("%d ", output->data.int8[i * 32]);
    // }
    printf("\n");
    count++;

    return cycle_count_after - cycle_count_before;
}
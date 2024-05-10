#ifndef _SPARSE_NNFUNCTIONS_H
#define _SPARSE_NNFUNCTIONS_H

#include "arm_math_types.h"
#include "arm_nn_types.h"
#include "sparse_nn_types.h"

#define USE_INTRINSIC

#ifdef __cplusplus
extern "C"
{
#endif

arm_status sparse_convolve_wrapper_s8(const cmsis_nn_context *ctx,
                                      const cmsis_nn_conv_params *conv_params,
                                      const cmsis_nn_per_channel_quant_params *quant_params,
                                      const cmsis_nn_dims *input_dims,
                                      const q7_t *input_data,
                                      const cmsis_nn_dims *filter_dims,
                                      const csr_conv_filters *csr_filters,
                                      const cmsis_nn_dims *bias_dims,
                                      const int32_t *bias_data,
                                      const cmsis_nn_dims *output_dims,
                                      q7_t *output_data);
                        
arm_status sparse_convolve_s8(const cmsis_nn_context *ctx,
                              const cmsis_nn_conv_params *conv_params,
                              const cmsis_nn_per_channel_quant_params *quant_params,
                              const cmsis_nn_dims *input_dims,
                              const q7_t *input_data,
                              const cmsis_nn_dims *filter_dims,
                              const csr_conv_filters *csr_filters,
                              const cmsis_nn_dims *bias_dims,
                              const int32_t *bias_data,
                              const cmsis_nn_dims *output_dims,
                              q7_t *output_data);

int32_t sparse_convolve_wrapper_s8_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                   const cmsis_nn_dims *input_dims,
                                                   const cmsis_nn_dims *filter_dims,
                                                   const cmsis_nn_dims *output_dims);

int32_t sparse_convolve_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims);

#ifdef __cplusplus
}
#endif

#endif // _SPARSE_NNFUNCTIONS_H
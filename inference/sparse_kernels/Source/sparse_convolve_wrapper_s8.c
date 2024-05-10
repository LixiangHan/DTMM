#include "sparse_nnfunctions.h"


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
                                      q7_t *output_data)
{
    return sparse_convolve_s8(ctx,
                              conv_params,
                              quant_params,
                              input_dims,
                              input_data,
                              filter_dims,
                              csr_filters,
                              bias_dims,
                              bias_data,
                              output_dims,
                              output_data);
}

int32_t sparse_convolve_wrapper_s8_get_buffer_size(const cmsis_nn_conv_params *conv_params,
                                                const cmsis_nn_dims *input_dims,
                                                const cmsis_nn_dims *filter_dims,
                                                const cmsis_nn_dims *output_dims)
{
    return sparse_convolve_s8_get_buffer_size(input_dims, filter_dims);
}
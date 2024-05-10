#ifndef _SPARSE_NNSUPPORTFUNCTIONS_H
#define _SPARSE_NNSUPPORTFUNCTIONS_H

#include "arm_math_types.h"
#include "arm_nn_types.h"
#include "sparse_nn_types.h"

#define USE_INTRINSIC

#ifdef __cplusplus
extern "C"
{
#endif

int8_t *sparse_nn_mat_mul_core_4x_s8(const int32_t row_elements,
                                     const int32_t input_ch,
                                     const int8_t *row_base,
                                     const csr_conv_filters *csr_filters,
                                     const int32_t out_ch,
                                     const cmsis_nn_conv_params *conv_params,
                                     const cmsis_nn_per_channel_quant_params *quant_params,
                                     const int32_t *bias,
                                     int8_t *output);


q7_t *sparse_nn_mat_mult_s8(const csr_conv_filters *csr_filters, // filters (out_channel * kernel_h * kernel_w * inchannel)
                            const q7_t *input_col, // lowered fearture map (4 * (kernel_h * kernel_w * in_channnel))
                            const uint16_t input_ch,
                            const uint16_t output_ch,
                            const uint16_t col_batches,
                            const int32_t *output_shift,
                            const int32_t *output_mult,
                            const int32_t out_offset,
                            const int32_t col_offset,
                            const int32_t row_offset, // 0
                            const int16_t activation_min,
                            const int16_t activation_max,
                            const uint16_t row_len, // number of elements of a filter
                            const int32_t *const bias,
                            q7_t *out);

#ifdef __cplusplus
}
#endif

#endif // _SPARSE_NNSUPPORTFUNCTIONS_H
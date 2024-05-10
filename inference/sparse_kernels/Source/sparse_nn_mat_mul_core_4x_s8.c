#include "sparse_nnsupportfunctions.h"
#include "arm_nnsupportfunctions.h"

int8_t *sparse_nn_mat_mul_core_4x_s8(const int32_t row_elements,
                                     const int32_t input_ch,
                                     const int8_t *row_base,
                                     const csr_conv_filters *csr_filters,
                                     const int32_t out_ch,
                                     const cmsis_nn_conv_params *conv_params,
                                     const cmsis_nn_per_channel_quant_params *quant_params,
                                     const int32_t *bias,
                                     int8_t *output)
{
    const int8_t *ip_row_base_0 = row_base;
    const int8_t *ip_row_base_1 = row_base + row_elements;
    const int8_t *ip_row_base_2 = row_base + (2 * row_elements);
    const int8_t *ip_row_base_3 = row_base + (3 * row_elements);

    int offset = 0;

    for (int i = 0; i < out_ch; i++)
    {
        int32_t acc_n0 = 0;
        int32_t acc_n1 = 0;
        int32_t acc_n2 = 0;
        int32_t acc_n3 = 0;

        int32_t sum_tmp = 0;
        for (int j = csr_filters->rowptr[i]; j < csr_filters->rowptr[i + 1]; j++)
        {
            const int8_t *col_base = csr_filters->filter_data + offset;
            offset += input_ch;

            uint32x4_t row_base = {(uint32_t)ip_row_base_3, (uint32_t)ip_row_base_2, (uint32_t)ip_row_base_1, (uint32_t)ip_row_base_0};

            __ASM volatile("    vldrw.u32       q5, [%[row_base]]       \n"
                           "    vadd.u32        q5, q5, %[offset]       \n"
                           "    vmov            r3, r1, q5[2], q5[0]    \n"
                           "    vldrb.8         q0, [%[col]], #16       \n"
                           "    vmov            r2, r0, q5[3], q5[1]    \n"
                           "    wlstp.8         lr, %[cnt], 1f          \n"
                           "2:                                          \n"
                           "    vaddva.s8       %[sum], q0              \n"
                           "    vldrb.8         q1, [r0], #16           \n"
                           "    vmladava.s8     %[out0], q0, q1         \n"
                           "    vldrb.8         q2, [r1], #16           \n"
                           "    vmladava.s8     %[out1], q0, q2         \n"
                           "    vldrb.8         q3, [r2], #16           \n"
                           "    vmladava.s8     %[out2], q0, q3         \n"
                           "    vldrb.8         q4, [r3], #16           \n"
                           "    vmladava.s8     %[out3], q0, q4         \n"
                           "    vldrb.8         q0, [%[col]], #16       \n"
                           "    letp            lr, 2b                  \n"
                           "1:                                          \n"
                           : [col] "+r"(col_base),
                             [sum] "+Te"(sum_tmp),
                             [out0] "+Te"(acc_n0),
                             [out1] "+Te"(acc_n1),
                             [out2] "+Te"(acc_n2),
                             [out3] "+Te"(acc_n3)
                           : [cnt] "r"(input_ch),
                             [row_base] "r"(&row_base),
                             [offset] "r"(csr_filters->col[j])
                           : "q0", "q1", "q2", "q3", "q4", "q5", "memory", "r0", "r1", "r2", "r3", "r14");
        }
        int32x4_t res = {acc_n0, acc_n1, acc_n2, acc_n3};
        sum_tmp *= conv_params->input_offset;
        if (bias)
        {
            sum_tmp += bias[i];
        }
        res = vaddq_n_s32(res, sum_tmp);

        res = arm_requantize_mve(res, quant_params->multiplier[i], quant_params->shift[i]);
        res = vaddq_n_s32(res, conv_params->output_offset);

        res = vmaxq_s32(res, vdupq_n_s32(conv_params->activation.min));
        res = vminq_s32(res, vdupq_n_s32(conv_params->activation.max));

        const uint32x4_t scatter_offset = {0, (uint32_t)out_ch, (uint32_t)out_ch * 2, (uint32_t)out_ch * 3};
        vstrbq_scatter_offset_s32(output, scatter_offset, res);
        output++;
    }

    return output + (3 * out_ch);
}
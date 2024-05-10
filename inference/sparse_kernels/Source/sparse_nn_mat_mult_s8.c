#include "sparse_nnsupportfunctions.h"
#include "arm_nnsupportfunctions.h"

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
                            q7_t *out)
{
    (void)row_offset;
    if (col_batches == 4)
    {
        int offset = 0;
        const int8_t *ip_c0_base = input_col;
        const int8_t *ip_c1_base = input_col + row_len;
        const int8_t *ip_c2_base = input_col + (2 * row_len);
        const int8_t *ip_c3_base = input_col + (3 * row_len);
        for (int i_out_ch = 0; i_out_ch < output_ch; i_out_ch++) // loop of filters
        {
            int32_t acc_0 = 0;
            int32_t acc_1 = 0;
            int32_t acc_2 = 0;
            int32_t acc_3 = 0;
            const int32_t row_loop_cnt = (input_ch + 7) / 8;
            for (int i_col = csr_filters->rowptr[i_out_ch]; i_col < csr_filters->rowptr[i_out_ch + 1]; i_col++)
            {
                const int8_t *ip_r0 = csr_filters->filter_data + offset;
                offset += input_ch;
                int32_t row_len_tmp = input_ch;
                const int8_t *ip_c0 = ip_c0_base + csr_filters->col[i_col];
                const int8_t *ip_c1 = ip_c1_base + csr_filters->col[i_col];
                const int8_t *ip_c2 = ip_c2_base + csr_filters->col[i_col];
                const int8_t *ip_c3 = ip_c3_base + csr_filters->col[i_col];

                for (int i_row_loop = 0; i_row_loop < row_loop_cnt; i_row_loop++)
                {
                    mve_pred16_t p = vctp16q((uint32_t)row_len_tmp);
                    const int16x8_t offset = vdupq_m_n_s16(vuninitializedq_s16(), col_offset, p);
                    row_len_tmp -= 8;

                    int16x8_t c0 = vldrbq_s16(ip_c0);
                    ip_c0 += 8;
                    c0 = vaddq_s16(c0, offset);

                    int16x8_t c1 = vldrbq_s16(ip_c1);
                    ip_c1 += 8;
                    c1 = vaddq_s16(c1, offset);

                    int16x8_t c2 = vldrbq_s16(ip_c2);
                    ip_c2 += 8;
                    c2 = vaddq_s16(c2, offset);

                    int16x8_t c3 = vldrbq_s16(ip_c3);
                    ip_c3 += 8;
                    c3 = vaddq_s16(c3, offset);

                    int16x8_t r0 = vldrbq_z_s16(ip_r0, p);
                    ip_r0 += 8;

                    acc_0 = vmladavaq_p_s16(acc_0, r0, c0, p);
                    acc_1 = vmladavaq_p_s16(acc_1, r0, c1, p);
                    acc_2 = vmladavaq_p_s16(acc_2, r0, c2, p);
                    acc_3 = vmladavaq_p_s16(acc_3, r0, c3, p);
                }
            }

            int32x4_t res = {acc_0, acc_1, acc_2, acc_3}; // combine result of 4 vector multiplication into a vector register
            if (bias)
            {
                res = vaddq_n_s32(res, bias[i_out_ch]); // add bias
            }
            res = arm_requantize_mve(res, output_mult[i_out_ch], output_shift[i_out_ch]); // requantized
            res = vaddq_n_s32(res, out_offset);                                           // add offset

            res = vmaxq_s32(res, vdupq_n_s32(activation_min)); // clamp
            res = vminq_s32(res, vdupq_n_s32(activation_max));

            const uint32x4_t scatter_offset = {0, output_ch, output_ch * 2, output_ch * 3}; // store format is height x width x channel
                                                                                            // the step length between two next elaments in a channel is 'output_ch'
            vstrbq_scatter_offset_s32(&out[i_out_ch], scatter_offset, res);                 // vstrbq_scatter_offset_s32(int8_t * base, uint32x4_t offset, int32x4_t value)                                                                       // store to memory from vector register with scatter offset
        }
        out += 4 * output_ch;
    }
    else
    {
        for (int i_col_batch = (col_batches & ~0x3); i_col_batch < (col_batches & 0x3); i_col_batch++)
        // 1 & ~0x3 = 0, 2 & ~0x3 = 0, 3 & ~0x3 = 0
        // 1 &  0x3 = 1, 2 &  0x3 = 2, 3 &  0x3 = 3
        {
            int offset = 0;
            const int8_t *ip_c0_base = input_col + (i_col_batch * row_len);
            for (int i_out_ch = 0; i_out_ch < output_ch; i_out_ch++)
            {
                int32_t acc_0 = 0;
                const int32_t row_loop_cnt = (input_ch + 7) / 8;
                for (int i_col = csr_filters->rowptr[i_out_ch]; i_col < csr_filters->rowptr[i_out_ch + 1]; i_col++)
                {
                    const int8_t *ip_r0 = csr_filters->filter_data + offset;
                    offset += input_ch;
                    int32_t row_len_tmp = input_ch;
                    const int8_t *ip_c0 = ip_c0_base + csr_filters->col[i_col];

                    for (int i_row_loop = 0; i_row_loop < row_loop_cnt; i_row_loop++)
                    {
                        const mve_pred16_t p = vctp16q((uint32_t)row_len_tmp);
                        const int16x8_t offset = vdupq_m_n_s16(vuninitializedq_s16(), col_offset, p);
                        row_len_tmp -= 8;

                        int16x8_t c0 = vldrbq_s16(ip_c0);
                        ip_c0 += 8;
                        c0 = vaddq_s16(c0, offset);

                        int16x8_t r0 = vldrbq_z_s16(ip_r0, p);
                        ip_r0 += 8;
                        acc_0 = vmladavaq_p_s16(acc_0, r0, c0, p);
                    }
                }

                if (bias)
                {
                    acc_0 += bias[i_out_ch];
                }
                acc_0 = arm_nn_requantize(acc_0, output_mult[i_out_ch], output_shift[i_out_ch]);
                acc_0 += out_offset;
                acc_0 = MAX(acc_0, activation_min);
                acc_0 = MIN(acc_0, activation_max);
                out[i_out_ch] = (q7_t)acc_0;
            }
            out += output_ch;
        }
    }
    return out;
}
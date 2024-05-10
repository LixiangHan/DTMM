#include "sparse_nnfunctions.h"
#include "sparse_nnsupportfunctions.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

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
							  q7_t *output_data)
{
	(void)bias_dims;

	if (ctx->buf == NULL && sparse_convolve_s8_get_buffer_size(input_dims, filter_dims) > 0)
	{
		return ARM_MATH_ARGUMENT_ERROR;
	}
	q7_t *buffer_a = (q7_t *)ctx->buf;

	const int32_t input_batches = input_dims->n;
	const uint16_t input_x = input_dims->w;
	const uint16_t input_y = input_dims->h;
	const uint16_t input_ch = input_dims->c;
	const uint16_t kernel_x = filter_dims->w;
	const uint16_t kernel_y = filter_dims->h;
	const uint16_t output_x = output_dims->w;
	const uint16_t output_y = output_dims->h;
	const uint16_t output_ch = output_dims->c;

	const uint16_t pad_x = conv_params->padding.w;
	const uint16_t pad_y = conv_params->padding.h;
	const uint16_t stride_x = conv_params->stride.w;
	const uint16_t stride_y = conv_params->stride.h;

	const int32_t input_offset = conv_params->input_offset;
	const int32_t out_offset = conv_params->output_offset;
	const int32_t out_activation_min = conv_params->activation.min;
	const int32_t out_activation_max = conv_params->activation.max;
	int32_t *output_mult = quant_params->multiplier;
	int32_t *output_shift = quant_params->shift;

	for (int i_batch = 0; i_batch < input_batches; i_batch++)
	{
		q7_t *out = output_data;
		q7_t *im2col_buf = (q7_t *)buffer_a;
		int32_t buffer_fill_cnt = 0;
		int32_t padded = 0;
		const int32_t num_elem = kernel_x * kernel_y * input_ch;
		for (int i_out_y = 0; i_out_y < output_y; i_out_y++)
		{
			for (int i_out_x = 0; i_out_x < output_x; i_out_x++)
			{
				const int base_idx_x = stride_x * i_out_x - pad_x;
				const int base_idx_y = stride_y * i_out_y - pad_y;
				for (int i_ker_y = 0; i_ker_y < kernel_y; i_ker_y++)
				{
					for (int i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
					{
						const int k_y = base_idx_y + i_ker_y;
						const int k_x = base_idx_x + i_ker_x;
						if (k_x < 0 || k_x >= input_x || k_y < 0 || k_y >= input_y)
						{
							memset(im2col_buf, (int8_t)-input_offset, sizeof(q7_t) * input_ch);
							padded = 0;
						}
						else
						{
							arm_memcpy_q7(im2col_buf, input_data + (k_y * input_x + k_x) * input_ch, input_ch);
						}
						im2col_buf += input_ch;
					}
				}
				buffer_fill_cnt++;

				if (buffer_fill_cnt == 4 && (padded == 0))
				{
					buffer_fill_cnt = 0;
					out = sparse_nn_mat_mul_core_4x_s8(num_elem,
													   input_ch,
													   (q7_t *)buffer_a,
													   csr_filters,
													   output_ch,
													   conv_params,
													   quant_params,
													   bias_data,
													   out);
					im2col_buf = (q7_t *)buffer_a;
				}
				else if (buffer_fill_cnt == 4 && (padded != 0))
				{
					buffer_fill_cnt = 0;
					out = sparse_nn_mat_mult_s8(csr_filters,
												(q7_t *)buffer_a,
												input_ch,
												output_ch,
												4,
												output_shift,
												output_mult,
												out_offset,
												input_offset,
												0,
												out_activation_min,
												out_activation_max,
												num_elem,
												bias_data,
												out);

					im2col_buf = (q7_t *)buffer_a;
					padded = 0;
				}
			}
		}
		if (buffer_fill_cnt != 0)
			{
				out = sparse_nn_mat_mult_s8(csr_filters,
											(q7_t *)buffer_a,
											input_ch,
											output_ch,
											buffer_fill_cnt,
											output_shift,
											output_mult,
											out_offset,
											input_offset,
											0,
											out_activation_min,
											out_activation_max,
											num_elem,
											bias_data,
											out);
			}
		input_data += (input_x * input_y * input_ch);
		output_data += (output_x * output_y * output_ch);
	}

	return ARM_MATH_SUCCESS;
}

int32_t sparse_convolve_s8_get_buffer_size(const cmsis_nn_dims *input_dims, const cmsis_nn_dims *filter_dims)
{
	return 4 * input_dims->c * filter_dims->h * filter_dims->w * (int32_t)sizeof(int8_t);
}
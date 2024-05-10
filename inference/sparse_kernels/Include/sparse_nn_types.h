#ifndef _SPARSE_NN_TYPES_H
#define _SPARSE_NN_TYPES_H

typedef struct
{
		const q7_t *filter_data;
		const q15_t *col;
		const q15_t *rowptr;
} csr_conv_filters;

#endif
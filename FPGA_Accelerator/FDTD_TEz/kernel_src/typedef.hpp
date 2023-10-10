#ifndef __TYPEDEF_HPP__
#define __TYPEDEF_HPP__


#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"
#include "assert.h"

#define N 400

typedef ap_fixed<16, 3, AP_RND_ZERO> d_htype;
typedef ap_ufixed<12, 1, AP_RND_ZERO> coe_htype;

typedef struct{
	d_htype Mx;
	d_htype My;
	d_htype Hz;
	coe_htype C;
	coe_htype Ca;
}point;

#endif

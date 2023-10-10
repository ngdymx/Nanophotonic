#include "typedef.hpp"

static void update_Hz_loop(d_htype Hz_new[], d_htype Hz[], d_htype My[], d_htype My_boundary[], d_htype Mx[], coe_htype C[]){
#pragma HLS INLINE
    for(int col = 0; col < N; col++){
#pragma HLS UNROLL
		if(col == N - 1){
			Hz_new[col] = C[col] * (Hz[col] - (My_boundary[col] - My[col]) + (0 - Mx[col]));
		}
		else{
			Hz_new[col] = C[col] * (Hz[col] - (My_boundary[col] - My[col]) + (Mx[col+1] - Mx[col]));
		}
    }
}

static void update_Mx_loop(d_htype Mx_new[], d_htype Mx[], d_htype Hz[], coe_htype C[], coe_htype Ca[]){
#pragma HLS INLINE
    for(int col = 0; col < N; col++){
#pragma HLS UNROLL
        if(col == 0){
            Mx_new[col] = C[col] * Mx[col] + Ca[col] * (Hz[col] - 0);
        }
        else{
            Mx_new[col] = C[col] * Mx[col] + Ca[col] * (Hz[col] - Hz[col - 1]);
        }
    }
}

static void update_My_loop(d_htype My_new[], d_htype My[], d_htype Hz[], d_htype Hz_boundary[], coe_htype C[], coe_htype Ca[]){
#pragma HLS INLINE
    for(int col = 0; col < N; col++){
#pragma HLS UNROLL
		My_new[col] = C[col] * My[col] - Ca[col] * (Hz[col] - Hz_boundary[col]);
    }
}

extern "C" {
void FDTD_Kernel(float out_f1[], float out_f2[], float source[], float C[][N], float Ca[][N],
		int T, int M, int src_row, int src_col,int det_f1_row, int det_f1_col,int det_f2_row, int det_f2_col){
#pragma HLS INTERFACE m_axi port = out_f1 offset = slave bundle = out_f1
#pragma HLS INTERFACE m_axi port = out_f2 offset = slave bundle = out_f2
#pragma HLS INTERFACE m_axi port = source offset = slave bundle = source_in
#pragma HLS INTERFACE m_axi port = C offset = slave bundle = C_in
#pragma HLS INTERFACE m_axi port = Ca offset = slave bundle = Ca_in
#pragma HLS INTERFACE s_axilite port = T
#pragma HLS INTERFACE s_axilite port = M
#pragma HLS INTERFACE s_axilite port = src_row
#pragma HLS INTERFACE s_axilite port = src_col
#pragma HLS INTERFACE s_axilite port = det_f1_row
#pragma HLS INTERFACE s_axilite port = det_f1_col
#pragma HLS INTERFACE s_axilite port = det_f2_row
#pragma HLS INTERFACE s_axilite port = det_f2_col
#pragma HLS INTERFACE s_axilite port = return
    static point space[4000][N];
#pragma HLS aggregate variable=space compact=bit
#pragma HLS BIND_STORAGE variable=space type=ram_2p impl=uram
#pragma HLS ARRAY_PARTITION variable=space type=complete dim=2
mem_rd_space:
	for(int row = 1; row < M+1; row++){
		for(int col = 0; col < N; col++){
#pragma HLS PIPELINE
			point temp;
			temp.Mx = 0;
			temp.My = 0;
			temp.Hz = 0;
			temp.C = C[row-1][col];
			temp.Ca = Ca[row-1][col];
			space[row][col] = temp;
		}
	}
	for(int col = 0; col < N; col++){
#pragma HLS PIPELINE
			point temp;
			temp.Mx = 0;
			temp.My = 0;
			temp.Hz = 0;
			temp.C = 0;
			temp.Ca = 0;
			space[0][col] = temp;
			space[M+1][col] = temp;
		}
update_loop:
    for(int t = 0; t < T; t++){
#pragma HLS PIPELINE off
    d_htype Hz_new_r0[N];
#pragma HLS ARRAY_PARTITION variable=Hz_new_r0 type=complete dim=0
    d_htype Hz_new_r1[N];
#pragma HLS ARRAY_PARTITION variable=Hz_new_r1 type=complete dim=0
    d_htype My_new[N];
#pragma HLS ARRAY_PARTITION variable=My_new type=complete dim=0
    d_htype Mx_new[N];
#pragma HLS ARRAY_PARTITION variable=Mx_new type=complete dim=0
    d_htype Hz_reg0[N];
#pragma HLS ARRAY_PARTITION variable=Hz_reg0 type=complete dim=0
    d_htype Mx_reg0[N];
#pragma HLS ARRAY_PARTITION variable=Mx_reg0 type=complete dim=0
    d_htype My_reg0[N];
#pragma HLS ARRAY_PARTITION variable=My_reg0 type=complete dim=0
    d_htype Hz_reg1[N];
#pragma HLS ARRAY_PARTITION variable=Hz_reg1 type=complete dim=0
    d_htype Mx_reg1[N];
#pragma HLS ARRAY_PARTITION variable=Mx_reg1 type=complete dim=0
    d_htype My_reg1[N];
#pragma HLS ARRAY_PARTITION variable=My_reg1 type=complete dim=0
    d_htype Hz_reg2[N];
#pragma HLS ARRAY_PARTITION variable=Hz_reg2 type=complete dim=0
    d_htype Mx_reg2[N];
#pragma HLS ARRAY_PARTITION variable=Mx_reg2 type=complete dim=0
    d_htype My_reg2[N];
#pragma HLS ARRAY_PARTITION variable=My_reg2 type=complete dim=0
    coe_htype C_reg0[N];
#pragma HLS ARRAY_PARTITION variable=C_reg0 type=complete dim=0
    coe_htype C_reg1[N];
#pragma HLS ARRAY_PARTITION variable=C_reg1 type=complete dim=0
    coe_htype C_reg2[N];
#pragma HLS ARRAY_PARTITION variable=C_reg2 type=complete dim=0
    coe_htype Ca_reg0[N];
#pragma HLS ARRAY_PARTITION variable=Ca_reg0 type=complete dim=0
    coe_htype Ca_reg1[N];
#pragma HLS ARRAY_PARTITION variable=Ca_reg1 type=complete dim=0
    coe_htype Ca_reg2[N];
#pragma HLS ARRAY_PARTITION variable=Ca_reg2 type=complete dim=0
    coe_htype C_reg3[N];
#pragma HLS ARRAY_PARTITION variable=C_reg3 type=complete dim=0
    coe_htype Ca_reg3[N];
#pragma HLS ARRAY_PARTITION variable=Ca_reg3 type=complete dim=0
Row_ite:
        for(int row = 0; row < M+6; row++){
#pragma HLS PIPELINE II=1
            if(row > 3){
Write_row:
                for(int col = 0; col < N; col++){
#pragma HLS UNROLL
                	point temp;
					temp.Mx = Mx_new[col];
					temp.My = My_new[col];
					temp.C = C_reg3[col];
					temp.Ca = Ca_reg3[col];
					if ((row == src_row + 4) && (col == src_col)){
						d_htype src_temp = (d_htype)source[t];
						temp.Hz = Hz_new_r1[col] + src_temp;
					}
					else{
						temp.Hz = Hz_new_r1[col];
					}
                	space[row-4][col] = temp;
                }
            }
            update_My_loop(My_new, My_reg2, Hz_new_r0, Hz_new_r1, C_reg2, Ca_reg2);
            update_Mx_loop(Mx_new, Mx_reg2, Hz_new_r0, C_reg2, Ca_reg2);
Hzx_ite:
            for(int col = 0; col < N; col++){
#pragma HLS UNROLL
                Hz_new_r1[col] = Hz_new_r0[col];
            }
            update_Hz_loop(Hz_new_r0, Hz_reg1, My_reg1, My_reg0, Mx_reg1, C_reg1);
            for(int col = 0; col < N; col++){
#pragma HLS UNROLL
				Mx_reg2[col] = Mx_reg1[col];
				My_reg2[col] = My_reg1[col];
				Hz_reg2[col] = Hz_reg1[col];
				Mx_reg1[col] = Mx_reg0[col];
				My_reg1[col] = My_reg0[col];
				Hz_reg1[col] = Hz_reg0[col];
				C_reg3[col] = C_reg2[col];
				Ca_reg3[col] = Ca_reg2[col];
				C_reg2[col] = C_reg1[col];
				Ca_reg2[col] = Ca_reg1[col];
				C_reg1[col] = C_reg0[col];
				Ca_reg1[col] = Ca_reg0[col];
            }
            if(row <= M+1){
Shift_reg:
				for(int col = 0; col < N; col++){
#pragma HLS UNROLL
					point r_temp = space[row][col];
					Mx_reg0[col] = r_temp.Mx;
					My_reg0[col] = r_temp.My;
					Hz_reg0[col] = r_temp.Hz;
					C_reg0[col] = r_temp.C;
					Ca_reg0[col] = r_temp.Ca;
				}
            }
		}//end row
        point oftemp1 = space[det_f1_row][det_f1_col];
        point oftemp2 = space[det_f2_row][det_f2_col];
        out_f1[t] = (float)(oftemp1.Hz);
        out_f2[t] = (float)(oftemp2.Hz);
    }//end T
}
}

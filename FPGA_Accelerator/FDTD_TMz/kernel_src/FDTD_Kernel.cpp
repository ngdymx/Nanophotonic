#include "typedef.hpp"

static void update_Mz_loop(d_htype Mz_new[], d_htype Mz[], d_htype Hy[], d_htype Hy_boundary[], d_htype Hx[], coe_htype C[], coe_htype Ca[]){
#pragma HLS INLINE
    for(int col = 0; col < N; col++){
#pragma HLS UNROLL
		if(col == 0){
			Mz_new[col] = C[col] * Mz[col] + Ca[col] * ((Hy[col] - Hy_boundary[col]) - (Hx[col] - 0));
		}
		else{
			Mz_new[col] = C[col] * Mz[col] + Ca[col] * ((Hy[col] - Hy_boundary[col]) - (Hx[col] - Hx[col - 1]));
		}
    }
}

static void update_Hx_loop(d_htype Hx_new[], d_htype Hx[], d_htype Mz[], coe_htype C[]){
#pragma HLS INLINE
    for(int col = 0; col < N; col++){
#pragma HLS UNROLL
        if(col == N - 1){
            Hx_new[col] = C[col] * (Hx[col] - (0 - Mz[col]));
        }
        else{
            Hx_new[col] = C[col] * (Hx[col] - (Mz[col + 1] - Mz[col]));
        }
    }
}

static void update_Hy_loop(d_htype Hy_new[], d_htype Hy[], d_htype Mz[], d_htype Mz_boundary[], coe_htype C[]){
#pragma HLS INLINE
    for(int col = 0; col < N; col++){
#pragma HLS UNROLL
		Hy_new[col] = C[col] * (Hy[col] + (Mz_boundary[col] - Mz[col]));
    }
}

extern "C" {
void FDTD_Kernel(float out_f1[], float out_f2[], float source[], float C[][N], float Ca[][N],
		int T, int M, int src_row, int src_col,int det_f1_row, int det_f1_col, int det_f2_row, int det_f2_col){
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
			temp.Hx = 0;
			temp.Hy = 0;
			temp.Mz = 0;
			temp.C = C[row-1][col];
			temp.Ca = Ca[row-1][col];
			space[row][col] = temp;
		}
	}
	for(int col = 0; col < N; col++){
#pragma HLS PIPELINE
		point temp;
		temp.Hx = 0;
		temp.Hy = 0;
		temp.Mz = 0;
		temp.C = 0;
		temp.Ca = 0;
		space[0][col] = temp;
		space[M+1][col] = temp;
	}
update_loop:
    for(int t = 0; t < T; t++){
#pragma HLS PIPELINE off
    d_htype Hx_new_r0[N];
#pragma HLS ARRAY_PARTITION variable=Hx_new_r0 type=complete dim=0
    d_htype Hx_new_r1[N];
#pragma HLS ARRAY_PARTITION variable=Hx_new_r1 type=complete dim=0
    d_htype Hx_new_r2[N];
#pragma HLS ARRAY_PARTITION variable=Hx_new_r2 type=complete dim=0
    d_htype Hy_new_r0[N];
#pragma HLS ARRAY_PARTITION variable=Hy_new_r0 type=complete dim=0
    d_htype Hy_new_r1[N];
#pragma HLS ARRAY_PARTITION variable=Hy_new_r1 type=complete dim=0
    d_htype Mz_new[N];
#pragma HLS ARRAY_PARTITION variable=Mz_new type=complete dim=0
    d_htype Mz_reg0[N];
#pragma HLS ARRAY_PARTITION variable=Mz_reg0 type=complete dim=0
    d_htype Hx_reg0[N];
#pragma HLS ARRAY_PARTITION variable=Hx_reg0 type=complete dim=0
    d_htype Hy_reg0[N];
#pragma HLS ARRAY_PARTITION variable=Hy_reg0 type=complete dim=0
    d_htype Mz_reg1[N];
#pragma HLS ARRAY_PARTITION variable=Mz_reg1 type=complete dim=0
    d_htype Hx_reg1[N];
#pragma HLS ARRAY_PARTITION variable=Hx_reg1 type=complete dim=0
    d_htype Hy_reg1[N];
#pragma HLS ARRAY_PARTITION variable=Hy_reg1 type=complete dim=0
    d_htype Mz_reg2[N];
#pragma HLS ARRAY_PARTITION variable=Mz_reg2 type=complete dim=0
    d_htype Hx_reg2[N];
#pragma HLS ARRAY_PARTITION variable=Hx_reg2 type=complete dim=0
    d_htype Hy_reg2[N];
#pragma HLS ARRAY_PARTITION variable=Hy_reg2 type=complete dim=0
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
					temp.Hx = Hx_new_r2[col];
					temp.Hy = Hy_new_r1[col];
					temp.C = C_reg3[col];
					temp.Ca = Ca_reg3[col];
					if ((row == src_row + 4) && (col == src_col)){
						d_htype src_temp = (d_htype)source[t];
						temp.Mz = Mz_new[col] + src_temp;
					}
					else{
						temp.Mz = Mz_new[col];
					}
                	space[row-4][col] = temp;
                }
            }
            update_Mz_loop(Mz_new, Mz_reg2, Hy_new_r0, Hy_new_r1, Hx_new_r1, C_reg2, Ca_reg2);
Hy_ite:
            for(int col = 0; col < N; col++){
#pragma HLS UNROLL
                Hy_new_r1[col] = Hy_new_r0[col];
            }
            update_Hy_loop(Hy_new_r0, Hy_reg1, Mz_reg1, Mz_reg0, C_reg1);
Hx_ite:
			for(int col = 0; col < N; col++){
#pragma HLS UNROLL
				Hx_new_r2[col] = Hx_new_r1[col];
				Hx_new_r1[col] = Hx_new_r0[col];
			}
			update_Hx_loop(Hx_new_r0, Hx_reg0, Mz_reg0, C_reg0);
            for(int col = 0; col < N; col++){
#pragma HLS UNROLL
				Hx_reg2[col] = Hx_reg1[col];
				Hy_reg2[col] = Hy_reg1[col];
				Mz_reg2[col] = Mz_reg1[col];
				Hx_reg1[col] = Hx_reg0[col];
				Hy_reg1[col] = Hy_reg0[col];
				Mz_reg1[col] = Mz_reg0[col];
				C_reg3[col] = C_reg2[col];
				Ca_reg3[col] = Ca_reg2[col];
				C_reg2[col] = C_reg1[col];
				Ca_reg2[col] = Ca_reg1[col];
				C_reg1[col] = C_reg0[col];
				Ca_reg1[col] = Ca_reg0[col];
            }
            if(row <= M + 1){
Shift_reg:
				for(int col = 0; col < N; col++){
#pragma HLS UNROLL
					point r_temp = space[row][col];
					Hx_reg0[col] = r_temp.Hx;
					Hy_reg0[col] = r_temp.Hy;
					Mz_reg0[col] = r_temp.Mz;
					C_reg0[col] = r_temp.C;
					Ca_reg0[col] = r_temp.Ca;
				}
            }
		}//end row
        point oftemp1 = space[det_f1_row][det_f1_col];
        point oftemp2 = space[det_f2_row][det_f2_col];
        out_f1[t] = (float)(oftemp1.Mz);
        out_f2[t] = (float)(oftemp2.Mz);
    }//end T
}
}
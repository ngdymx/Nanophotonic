#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define M 20
#define N 20
#define T 80
int main(int argc, char** argv) {
	FILE* fp;
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
	const int T_val = T;
	const int M_val = M;
	printf("[Process]\tEmulation starts!\n");
    printf("[Info.]\tTotal time steps:\t%d\n", T);
    printf("[Info.]\tUse DC source for test!\n");

    printf("[Process]\tStart allocating vectors...\n");
    float source[T];
    float C[M][N];
    float Ca[M][N];
    float out_f1[T];
    float out_f2[T];
	int src_row = 11;
	int src_col = 11;
	int det_f1_row = 16;
	int det_f1_col = 16;
	int det_f2_row = 16;
	int det_f2_col = 16;

    printf("[Process]\tFinish allocating vectors!\n");

	printf("[Process]\tStart loading matrixes and vectors...\n");
    fp = fopen("/home/michelle/Projects/Matlab/FDTD/C.txt","r");
	if (fp == NULL){
		printf("[Error]\tFaild to load txt file!\n");
		printf("[Error]\tBreak simulation!\n");
		return 0 ;
    }

	printf("[Process]\tReading C...\n");
    for(int i = 0;i < M;i++){
    	for(int j = 0; j < N; j++){
//    		fscanf(fp,"%f",&C[i][j]);
    		C[i][j] = 1;
			printf("%.4f\n",C[i][j]);
    	}
	}
	fclose(fp);

    fp = fopen("/home/michelle/Projects/Matlab/FDTD/Ca.txt","r");
	printf("[Process]\tReading Ca...\n");
    for(int i = 0;i < M;i++){
    	for(int j = 0; j < N; j++){
//    		fscanf(fp,"%f",&Ca[i][j]);
    		Ca[i][j] = 0.4901;
			printf("%.4f\n",Ca[i][j]);
    	}
	}
	fclose(fp);

	printf("[Process]\tReading source...\n");
	for(int t = 0; t < T; t++){
		source[t] = sin(2 * 3.14159 * (t+1) * 0.0115);
		printf("%.4f\n",source[t]);
	}

    printf("[Process]\tSoftware initialization done!\n\n");
    printf("[Process]\tRun hardware initialization...\n\n");

    int device_id = 0;
	std::cout << "\tOpen the device" << device_id << std::endl;
	auto device = xrt::device(device_id);
	std::cout << "\tLoad the xclbin " << binaryFile << std::endl;
	auto uuid = device.load_xclbin(binaryFile);

	// Connect to kernels
	printf("[Process]\tConnect to %s kernel...\n", "FDTD_Kernel");
	auto FDTD_Kernel = xrt::kernel(device, uuid, "FDTD_Kernel");

	//allocate buffer
	std::cout << "[Process]\tAllocate Buffer in Global Memory...\n";
	printf("[Process]\tAllocate %s buffer...\n","out_f1");
	auto bo_out_f1 = xrt::bo(device, sizeof(out_f1), FDTD_Kernel.group_id(0));
	printf("[Process]\tAllocate %s buffer...\n","out_f2");
	auto bo_out_f2 = xrt::bo(device, sizeof(out_f2), FDTD_Kernel.group_id(1));
	printf("[Process]\tAllocate %s buffer...\n","source");
	auto bo_src = xrt::bo(device, sizeof(source), FDTD_Kernel.group_id(2));
	printf("[Process]\tAllocate %s buffer...\n","C");
	auto bo_C = xrt::bo(device, sizeof(C), FDTD_Kernel.group_id(3));
	printf("[Process]\tAllocate %s buffer...\n","Ca");
	auto bo_Ca = xrt::bo(device, sizeof(Ca), FDTD_Kernel.group_id(4));
    // Allocate Buffer in Global Memory

	bo_src.write(source);
	bo_src.sync(XCL_BO_SYNC_BO_TO_DEVICE);

	bo_C.write(C);
	bo_C.sync(XCL_BO_SYNC_BO_TO_DEVICE);

	bo_Ca.write(Ca);
	bo_Ca.sync(XCL_BO_SYNC_BO_TO_DEVICE);

	auto FDTD_Kernel_run = xrt::run(FDTD_Kernel);

	FDTD_Kernel_run.set_arg(0, bo_out_f1);
	FDTD_Kernel_run.set_arg(1, bo_out_f2);
	FDTD_Kernel_run.set_arg(2, bo_src);
	FDTD_Kernel_run.set_arg(3, bo_C);
	FDTD_Kernel_run.set_arg(4, bo_Ca);
	FDTD_Kernel_run.set_arg(5, T_val);
	FDTD_Kernel_run.set_arg(6, M_val);
	FDTD_Kernel_run.set_arg(7, src_row);
	FDTD_Kernel_run.set_arg(8, src_col);
	FDTD_Kernel_run.set_arg(9, det_f1_row);
	FDTD_Kernel_run.set_arg(10, det_f1_col);
	FDTD_Kernel_run.set_arg(11, det_f2_row);
	FDTD_Kernel_run.set_arg(12, det_f2_col);

    printf("[Process]\tHardware initialization done!\n\n");
    printf("[Process]\tStart writing matrixes to hardware...\n");

    std::cout << "[Process]\tFinished writing!" << std::endl;
    printf("[Process]\tStart simulation...\n");
    for (int tt = 0; tt < 2; tt++){
		printf("\tRound %d...",tt+1);
		FDTD_Kernel_run.start();

		// wait
		FDTD_Kernel_run.wait();
		std::cout << "[Process]\t\tupdate finished!" << std::endl;
		std::cout << "[Process]\t\tback finished!" << std::endl;
		std::cout << "[Process]\t\tobserver finished!" << std::endl;
		std::cout << "[Process]\tFinished one round!" << std::endl;

		bo_out_f1.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		bo_out_f1.read(out_f1);
		bo_out_f2.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
		bo_out_f2.read(out_f2);


		usleep(10000);
		char fname[256];
		sprintf(fname, "/home/michelle/Projects/Vitis/FDTD_Vitis/Test/FDTD_S2_make/res_%d.txt", tt);
		fp = fopen(fname,"w");
		if (fp == NULL){
			printf("[Error]\tCannot open res.txt! Break!\n");
			return 0;
		}
		for (int t = 0; t < T; t++){
			fprintf(fp,"%.4f\t",out_f1[t]);
			fprintf(fp,"%.4f\t",out_f2[t]);
			fprintf(fp,"\n");
			out_f1[t] = -1;
			out_f2[t] = -1;
		}
		fclose(fp);
	}
    std::cout << "[Process]\tFinished simulation!" << std::endl;


    printf("[Process]\tWriting final result to res.txt!\n");



	printf("[Process]\tHost program finished!\n");


    return EXIT_SUCCESS;
}
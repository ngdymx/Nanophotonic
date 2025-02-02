################################################################################
# this file was created by alfred. Do not trust it.
#
# tool chain path
XILINX_VITIS ?= /tools/Xilinx/Vitis/2021.2
XILINX_XRT ?= /opt/xilinx/xrt
XILINX_VIVADO ?= /tools/Xilinx/Vivado/2021.2
XILINX_HLS_INCLUDES ?= /tools/Xilinx/Vitis_HLS/2021.2/include

# Platform path
VITIS_PLATFORM = xilinx_aws-vu9p-f1_shell-v04261818_201920_3
VITIS_PLATFORM_DIR = /home/alfred/Projects/Vitis/aws-fpga/Vitis/aws_platform/xilinx_aws-vu9p-f1_shell-v04261818_201920_3
VITIS_PLATFORM_PATH = $(VITIS_PLATFORM_DIR)/xilinx_aws-vu9p-f1_shell-v04261818_201920_3.xpfm

# Project Name
PRJ_NAME ?= FDTD
TARGET ?= hw_emu
# Host source files
HOST_SRC_DIR ?= host_src
HOST_SRC += $(HOST_SRC_DIR)/host.cpp

# Kernel Source Path, do not modify
KERNEL_O_DIR ?= build/vitis_hls
KERNEL_SRC_DIR ?= kernel_src

# Kernel to build
HW_KERNEL_OBJS += $(KERNEL_O_DIR)/FDTD_Kernel.xo

# set kernel frequency if necessary
KERNEL_EXT_CONFIG += --kernel_frequency
KERNEL_EXT_CONFIG += 125

# Bit Container Linker Configuration, edit independently
BC_CONFIG += $(KERNEL_SRC_DIR)/linker.cfg

################################################################################
################# do not modify contents below if not necessary ################
################################################################################
VPP_OPTS = --target $(TARGET)

# compiler tools
VPP_LINKER ?= ${XILINX_VITIS}/bin/v++
VPP ?= ${XILINX_VITIS}/bin/v++
HOST_CXX ?= g++
RM = rm -f
RMDIR = rm -rf
#
# host complie flags
CXXFLAGS += -std=c++1y -DVITIS_PLATFORM=$(VITIS_PLATFORM) -D__USE_XOPEN2K8 -I$(XILINX_XRT)/include/ -I$(XILINX_HLS_INCLUDES)/ -O2 -g -Wall -c -fmessage-length=0
LDFLAGS += -luuid -lxrt_coreutil -lxilinxopencl -lpthread -lrt -lstdc++ -L$(XILINX_XRT)/lib/ -Wl,-rpath-link,$(XILINX_XRT)/lib

BUILD_SUBDIRS += build

BINARY_CONTAINER += build/vivado/bit_container.xclbin

HOST_OBJ +=	build/host/host.o 
HOST_EXE = build/host/$(PRJ_NAME)

LOG_DIR ?= build/logs
REPORT_DIR ?= build/reports

RUN_EXE ?= $(PRJ_NAME)
RUN_BIN ?= bit_container.xclbin


KERNEL_EXT_CONFIG += --platform
KERNEL_EXT_CONFIG += $(VITIS_PLATFORM_PATH)
KERNEL_EXT_CONFIG += --temp_dir
KERNEL_EXT_CONFIG += build/vitis_hls
KERNEL_EXT_CONFIG += --log_dir
KERNEL_EXT_CONFIG += $(LOG_DIR)
KERNEL_EXT_CONFIG += --report_dir
KERNEL_EXT_CONFIG += $(REPORT_DIR)
KERNEL_EXT_CONFIG += -s
KERNEL_EXT_CONFIG += -g


LINK_EXT_CONFIG += --temp_dir
LINK_EXT_CONFIG += build/vivado

#
# primary build targets
#

.PHONY: all clean host kernel link run
all: $(HW_KERNEL_OBJS) $(BINARY_CONTAINER) $(HOST_EXE)
	-@cp $(HOST_EXE) ./
	-@cp $(BINARY_CONTAINER) ./

host: $(HOST_EXE)


kernel: $(HW_KERNEL_OBJS)


link: $(BINARY_CONTAINER)


run: $(RUN_EXE) $(RUN_BIN) emconfig.json
	export XCL_EMULATION_MODE=$(TARGET) && ./$(RUN_EXE) $(RUN_BIN)

$(RUN_EXE): $(HOST_EXE)
	-@cp $(HOST_EXE) ./

$(RUN_BIN): $(BINARY_CONTAINER)
	-@cp $(BINARY_CONTAINER) ./

clean:
	-$(RM) $(HW_KERNEL_OBJS) $(BINARY_CONTAINER) *.log *.mdb 
	-$(RMDIR) $(BUILD_SUBDIRS)
	-$(RMDIR) .Xil
	-$(RM) *.json
	-$(RM) $(RUN_EXE)
	-$(RM) $(RUN_BIN)
	-$(RMDIR) .run
	-$(RMDIR) .ipcache

.PHONY: incremental
incremental: all


nothing:

# build kernels first, automatic build all kernels
$(KERNEL_O_DIR)/%.xo: $(KERNEL_SRC_DIR)/%.cpp
	-@mkdir -p $(@D)
	-@mkdir -p $(LOG_DIR)
	-@mkdir -p $(REPORT_DIR)
	-@$(RM) $@
	$(VPP) $(VPP_OPTS) --compile -I"$(<D)" --kernel $(basename $(@F)) --advanced.misc solution_name=$(basename $(@F)) $(KERNEL_EXT_CONFIG) -o"$@" "$<"

	
# link project secondly
$(BINARY_CONTAINER): $(HW_KERNEL_OBJS) $(BC_CONFIG)
	-@mkdir -p $(@D)
	$(VPP_LINKER) $(VPP_OPTS) --link --platform $(VITIS_PLATFORM_PATH) --config $(BC_CONFIG)  --temp_dir build/vivado --log_dir $(LOG_DIR) --report_dir $(REPORT_DIR) -R2 -o"$@" $(HW_KERNEL_OBJS)


# bulid host finally
$(HOST_EXE): $(HOST_OBJ)
	-@mkdir -p $(@D)
	$(HOST_CXX) -o "$@" $(+) $(LDFLAGS)

$(HOST_OBJ): $(HOST_SRC)
	-@mkdir -p $(@D)
	$(HOST_CXX) $(CXXFLAGS) -o "$@" "$<"

emconfig.json:
	-@emconfigutil --platform $(VITIS_PLATFORM_PATH)


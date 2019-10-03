# mkfcu.make
UNAME := $(shell uname -a)
NVCCPATH := $(shell command -v nvcc 2>/dev/null)
#CUDA_PATH := $(shell "echo $CUDA_PATH")
CUDA_PATH ?= /usr/local/cuda
CULBPATH := $(CUDA_PATH)/lib64

# GNU stuff
GOPT := -Os 
CC := gcc -std=c11
COPT := $(GOPT)
#99
CXX := g++ -std=c++11
CXOPT := $(GOPT)


NVCC := nvcc -arch=sm_50
# -gencode=arch=compute_50,code=sm_50
NVOPT := -O3
#NVOPT := -O0 -g -G 
# -dc ???
## NVCC is unfussy regarding standards compliance
# -std=c++11
# -std=c99
# -std=gnu99

TARGET := mkfcu
MAKEFILE := $(TARGET).make

SRC_DIR := src
HDR_DIR := $(SRC_DIR)
OBJ_DIR := obj
CMN_DIR := ../Common/src
INC_DIR := inc

SRC := $(SRC_DIR)/mkfCUDA.cu $(SRC_DIR)/binMapCUDA.cu $(SRC_DIR)/ctUtil.cu
HDR := $(HDR_DIR)/mkfCUDA.h $(SRC_DIR)/binMapCUDA.h $(HDR_DIR)/ctUtil.h
OBJ := $(OBJ_DIR)/mkfUtil.o $(OBJ_DIR)/mkfHacks.o $(OBJ_DIR)/cell8Sym.o $(OBJ_DIR)/binMapUtil.o $(OBJ_DIR)/geomHacks.o $(OBJ_DIR)/geomRaster.o $(OBJ_DIR)/geomObj.o $(OBJ_DIR)/geomSVU.o $(OBJ_DIR)/encoding.o $(OBJ_DIR)/util.o $(OBJ_DIR)/report.o


LIBDEF := -lm -L$(CULBPATH) -lcudart
INCDEF := -I$(CMN_DIR) -I$(INC_DIR) -DMKF_CUDA_MAIN

### Rebuild any GNU modules to objects as necessary

# Move any object files to the expected location
$(OBJ_DIR)/%.o : %.o
	mv $< $@

%.o : $(CMN_DIR)/%.c
	$(CC) $(COPT) $(INCDEF) $< -c

%.o : $(SRC_DIR)/%.c
	$(CC) $(COPT) $(INCDEF) $< -c

%.o : $(SRC_DIR)/%.cpp
	$(CXX) $(CXOPT) $(INCDEF) $< -c

### Use NVCC source build for CUDA stuff (avoids tedious linker issues) 

$(TARGET) : $(SRC) $(HDR) $(OBJ) $(MAKEFILE)
	$(NVCC) $(NVOPT) $(INCDEF) $(SRC) $(OBJ) $(LIBDEF) -o $@


.PHONY : all clean run

all : clean run

run : $(TARGET)
	./$<

clean :
	rm -f $(TARGET) $(OBJ)

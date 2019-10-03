# mkft.make
UNAME := $(shell uname -a)
PGCCPATH := $(shell command -v pgcc 2>/dev/null)
NVCCPATH := $(shell command -v nvcc 2>/dev/null)
#PGI_PATH := $(shell "echo $PGI_PATH")
PGI_PATH ?= /opt/pgi/linux86-64/2019
CULBPATH := $(PGI_PATH)/cuda/10.1/lib64

LIBDEF := -lm

ifdef NVCCPATH
NVCC := nvcc
NVOPT := -arch=sm_50 -ccbin=pgc++ 
#NVOPT += -g -G
NVOPT += -O3
endif

ifdef PGCCPATH
BUILD := NCRMNTL
CC := pgcc
CCPP := pgc++ -std=c++11
LIBDEF += -lstdc++
ACC := -Mautoinline -acc=verystrict -ta=multicore,tesla
# -Minfo=all
# multicore,tesla:cc50

else # Default compiler gcc / clang, assume no ACC
#BUILD := FLLSRC
CC := gcc -Wall
CCPP := g++
LIBDEF += -lstdc++
OPT := -march=native
# problematic...
# -std=c11 -D__USE_MISC -pedantic
# defaults
#CC := clang -Wall # Code gen errors ?
# -Os O2_MN_SZ
# -Oz SIZE
endif
ACC ?=

OPT += -O3
#OPT += -g -O0

TARGET := mkft
MAKEFILE := $(TARGET).make
BUILD ?= INCREMENTAL

SRC_DIR := src
HDR_DIR := $(SRC_DIR)
OBJ_DIR := obj
CMN_DIR := ../Common/src
INC_DIR := inc

C_SRC := $(shell ls $(SRC_DIR)/*.c)
CU_SRC := $(shell ls $(SRC_DIR)/*.cu)
HDR := $(shell ls $(SRC_DIR)/*.h)
CPP_SRC := $(shell ls $(SRC_DIR)/*.cpp)
CPP_HDR := $(shell ls $(SRC_DIR)/*.hpp)

C_OBJ := $(C_SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
CU_OBJ := $(CU_SRC:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CPP_OBJ := $(CPP_SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

CMN_SRC := $(shell ls $(CMN_DIR)/*.c)
CMN_OBJ := $(CMN_SRC:$(CMN_DIR)/%.c=$(OBJ_DIR)/%.o)
INCDEF := -I$(CMN_DIR) -I$(INC_DIR)

OBJ := $(C_OBJ) $(CMN_OBJ)

# Move any object files to the expected location
$(OBJ_DIR)/%.o : %.o
	mv $< $@

ifdef NVCC

OBJ += $(CU_OBJ)

LIBDEF += -L$(CULBPATH) -lcudart

INCDEF += -DMKF_CUDA
INCDEF += -DMKF_ACC_CUDA_INTEROP

%.o : $(SRC_DIR)/%.cu $(HDR_DIR)/%.h
	$(NVCC) $(NVOPT) $(INCDEF) $< -c

endif

#ifdef $(CPP_OBJ)

OBJ +=  $(CPP_OBJ)

# $(HDR_DIR)/%.hpp
%.o : $(SRC_DIR)/%.cpp $(CPP_HDR)
	$(CCPP) $(OPT) $(INCDEF) $< -c

#endif

ifeq ($(BUILD),FLLSRC)
# Full build from source every time : not workable with multiple compilers
# reliable with pgcc+nvcc multi-compiler...

$(TARGET) : $(C_SRC) $(CMN_SRC) $(HDR) $(MAKEFILE) $(CPP_OBJ)
	$(CC) $(OPT) $(ACC) $(INCDEF) $(C_SRC) $(CMN_SRC) $(LIBDEF) $(CPP_OBJ) -o $@
	#$(CUCC) $(OPT) $(INCDEF) $(CU_SRC) -c

else # Build incrementally if necessary

%.o : $(SRC_DIR)/%.c $(HDR_DIR)/%.h
	$(CC) $(OPT) $(ACC) $(INCDEF) $< -c

%.o : $(CMN_DIR)/%.c $(CMN_DIR)/%.h
	$(CC) $(OPT) $(INCDEF) $< -c

$(TARGET) : $(OBJ) $(MAKEFILE)
	$(CC) $(OPT) $(ACC) $(LIBDEF) $(OBJ) -o $@

endif # ifeq($(BUILD) ...

.PHONY : all clean run

all : clean run

run : $(TARGET)
	./$<

clean :
	rm -f $(TARGET) $(OBJ)

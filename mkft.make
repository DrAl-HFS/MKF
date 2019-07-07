# mkft.make
UNAME := $(shell uname -a)
PGCCPATH := $(shell command -v pgcc 2>/dev/null)
LCUPATH := /opt/pgi/linux86-64/2019/cuda/10.1/lib64
# $(shell echo $(PGI_PATH))/cuda/10.1/lib64 ???
# $PGI_CUDA_LIB_PATH

###ifneq (,$(findstring /bin/pgcc,$(PGCCPATH)))
ifdef PGCCPATH
BUILD := NCRMNTL
CC := pgcc
CUCC := nvcc
OPT := -g
#-O2
ACC := -Mautoinline -acc=verystrict -ta=multicore
# -Minfo=all
# multicore,tesla
else
BUILD := FLLSRC
CC := gcc
OPT := -Wall -Os
# full debug...
#OPT := -Wall -g -O0
# problematic...
# -std=c11 -D__USE_MISC -pedantic
# defaults
#CC := clang  # Code gen errors ?
#OPT := -Wall -Oz
endif
ACC ?= 

TARGET := mkft
MAKEFILE := $(TARGET).make
BUILD ?= FLLSRC

SRC_DIR := src
HDR_DIR := $(SRC_DIR)
OBJ_DIR := obj
CMN_DIR := ../Common/src
INC_DIR := inc

C_SRC := $(shell ls $(SRC_DIR)/*.c)
CU_SRC := $(shell ls $(SRC_DIR)/*.cu)
HDR := $(shell ls $(SRC_DIR)/*.h)
C_OBJ := $(C_SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
CU_OBJ := $(CU_SRC:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

CMN_SRC := $(shell ls $(CMN_DIR)/*.c)
CMN_OBJ := $(CMN_SRC:$(CMN_DIR)/%.c=$(OBJ_DIR)/%.o)
LIBDEF := -lm -lcudart -L$(LCUPATH)
INCDEF := -I$(CMN_DIR) -I$(INC_DIR)
#-DMK_

OBJ := $(C_OBJ) $(CMN_OBJ) $(CU_OBJ)

# Move any object files to the expected location
$(OBJ_DIR)/%.o : %.o
	mv $< $@

%.o : $(SRC_DIR)/%.cu $(HDR_DIR)/%.h
	$(CUCC) $(OPT) $(INCDEF) $< -c


ifeq ($(BUILD),FLLSRC)
# Full build from source every time : not reliable with pgcc+nvcc multi-compiler...
$(TARGET) : $(C_SRC) $(CMN_SRC) $(HDR) $(MAKEFILE) $(CU_OBJ) 
	$(CC) $(OPT) $(ACC) $(INCDEF) $(C_SRC) $(CMN_SRC) $(LIBDEF) $(CU_OBJ) -o $@
	#$(CUCC) $(OPT) $(INCDEF) $(CU_SRC) -c

else # Build incrementally if necessary

%.o : $(SRC_DIR)/%.c $(HDR_DIR)/%.h
	$(CC) $(OPT) $(ACC) $(INCDEF) $(DEFS) $< -c

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

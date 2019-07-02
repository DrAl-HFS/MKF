# mkft.make
UNAME := $(shell uname -a)
PGCCOUT := $(shell pgcc 2>&1)

#ifeq ($(PGCCOUT),pgcc)
ifneq (,$(findstring pgcc,$(PGCCOUT)))
BUILD := NCRMNTL
CC := pgcc
OPT := -O2
ACC := -Mautoinline -acc=verystrict -ta=tesla -Minfo=all
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

SRC := $(shell ls $(SRC_DIR)/*.c)
HDR := $(shell ls $(SRC_DIR)/*.h)
OBJ := $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
CMN_SRC := $(shell ls $(CMN_DIR)/*.c)
CMN_OBJ := $(CMN_SRC:$(CMN_DIR)/%.c=$(OBJ_DIR)/%.o)
LIBS := -lm
PATHS := -I$(CMN_DIR) -I$(INC_DIR)
DEFS :=
#-DMK_

# Move any object files to the expected location
$(OBJ_DIR)/%.o : %.o
	mv $< $@

ifeq ($(BUILD),FLLSRC)
# Full build from source every time
$(TARGET) : $(SRC) $(CMN_SRC) $(HDR) $(MAKEFILE)
	$(CC) $(OPT) $(ACC) $(PATHS) $(DEFS) $(LIBS) $(SRC) $(CMN_SRC) -o $@

else # Build incrementally if efficiency becomes a concern...
%.o : $(SRC_DIR)/%.c $(HDR_DIR)/%.h
	$(CC) $(OPT) $(ACC) $(PATHS) $(DEFS) $< -c

%.o : $(CMN_DIR)/%.c $(CMN_DIR)/%.h
	$(CC) $(OPT) $(PATHS) $(DEFS) $< -c

$(TARGET) : $(OBJ) $(CMN_OBJ) $(MAKEFILE)
	$(CC) $(OPT) $(ACC) $(LIBS) $(OBJ) $(CMN_OBJ) -o $@

endif # ifeq($(BUILD) ...

.PHONY : all clean run

all : clean run

run : $(TARGET)
	./$<

clean :
	rm -f $(TARGET) $(OBJ)

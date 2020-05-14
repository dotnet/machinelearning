CXX = g++
CXXFLAGS = -Wall -O3 -pthread -std=c++0x -march=native
OMPFLAG = -fopenmp
SHVER = 2

# run `make clean all' if you change the following flags.

# comment the following flag if you want to disable SSE or enable AVX
DFLAG = -DUSESSE

# uncomment the following flags if you want to use AVX
#DFLAG = -DUSEAVX
#CXXFLAGS += -mavx

# uncomment the following flags if you do not want to use OpenMP
DFLAG += -DUSEOMP
CXXFLAGS += $(OMPFLAG)

KERNEL_NAME = $(shell uname -s)

ifeq ($(KERNEL_NAME), Darwin)
	LIB_EXT = $(SHVER).dylib
	SO_NAME = -install_name
else
	LIB_EXT = so.$(SHVER)
	SO_NAME = -soname
endif

all: mf-train mf-predict

lib: mf.o
	$(CXX) -shared -Wl,$(SO_NAME),libmf.$(LIB_EXT) -o libmf.$(LIB_EXT) mf.o

mf-train: mf-train.cpp mf.o
	$(CXX) $(CXXFLAGS) $(DFLAG) -o $@ $^

mf-predict: mf-predict.cpp mf.o
	$(CXX) $(CXXFLAGS) $(DFLAG) -o $@ $^

mf.o: mf.cpp mf.h
	$(CXX) $(CXXFLAGS) $(DFLAG) -c -fPIC -o $@ $<

clean:
	rm -f mf-train mf-predict mf.o libmf.$(LIB_EXT)

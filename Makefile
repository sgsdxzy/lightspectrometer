CC=gcc
CXX=g++
RM=rm -f
CFLAGS=-march=native -O3 -pipe -fopenmp
CXXFLAGS=$(CFLAGS)
LDFLAGS=-Wl,-O1,--sort-common,--as-needed,-z,relro
LDLIBS=-lboost_serialization

COMMON_SOURCES = edpsolver.cpp magnet.cpp spectrometer.cpp
COMMON_OBJECTS = $(COMMON_SOURCES:.cpp=.o)

all:simple_output get_solver_data solvertest

simple_output:$(COMMON_OBJECTS) simple_output.cpp
	$(CXX) $(CFLAGS) $(LDLIBS)  $^ -o $@ $(LDFLAGS) 

get_solver_data:$(COMMON_OBJECTS) get_solver_data.cpp
	$(CXX) $(CFLAGS) $(LDLIBS)  $^ -o $@ $(LDFLAGS) 

solvertest:$(COMMON_OBJECTS) solvertest.cpp
	$(CXX) $(CFLAGS) $(LDLIBS)  $^ -o $@ $(LDFLAGS) 

%.o: %.cpp 
	$(CXX) $(CFLAGS) $(LDLIBS) -c $< $(LDFLAGS) 

clean:
	$(RM) $(COMMON_OBJECTS) simple_output get_solver_data solvertest

CXX        = nvcc
LIB        = -L$(OPENCL_LIBDIR) -lOpenCL
CXXFLAGS   = -O3 -DWITH_FLOATS=0 --std=c++11

INCLUDES    += -I ../include
GPU_OPTS   = -D lgWARP=5

SOURCES_CPP =ProjectMain.cu ProjHelperFun.cu ProjCoreOrig.cu kernels.cu
HELPERS     =ProhHelperFun.h ../include/Constants.h ../include/ParseInput.h ../include/ParserC.h
#OBJECTS     =ProjectMain.o ProjHelperFun.o  ProjCoreOrig.o
EXECUTABLE  =runproject


default: gpu

.cu.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) $(CXXFLAGS) $(GPU_OPTS) $(INCLUDES) -c $@ $<


gpu: $(EXECUTABLE)
$(EXECUTABLE):
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXECUTABLE) ProjectMain.cu


run_small: $(EXECUTABLE)
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_medium: $(EXECUTABLE)
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_large: $(EXECUTABLE)
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE) 2> Debug.txt

clean:
	rm -f Debug.txt $(EXECUTABLE) $(OBJECTS)


profile_small: $(EXECUTABLE)
	nvprof --profile-child-processes --analysis-metrics -o analysis%p.nvprof make run_small

profile_medium: $(EXECUTABLE)
	nvprof --profile-child-processes --analysis-metrics -o analysis%p.nvprof make run_medium

profile_large: $(EXECUTABLE)
	nvprof --profile-child-processes --analysis-metrics -o analysis%p.nvprof make run_large

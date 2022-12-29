

cu  := $(shell find src -name "*.cu")
cuo := $(patsubst src/%.cu,objs/%.cuo,$(cu))


include_paths := /usr/local/cuda-11/include 

library_paths := /usr/local/cuda-11/lib64

link_library  := cudart  cublas \
                 pthread

I = $(include_paths:%=-I%)
l = $(link_library:%=-l%)
L = $(library_paths:%=-L%)
r = $(library_paths:%=-Wl,-rpath=%)

compile_options := -m64 -std=c++11 -g -O3 -w\
				   -Xcompiler -fpic -Xcompiler -fopenmp \
				   $(I)

linking_options := $(l) $(L) $(r)

objs/%.cuo : src/%.cu
	@echo Compile $<
	@mkdir -p $(dir $@)
	@nvcc -c $< -o $@ $(compile_options)

workspace/exec : $(cuo)
	@echo Linking $@
	@mkdir -p $(dir $@)
	@g++ $^ -o $@ $(linking_options)

run : workspace/exec
	@./$<

clean :
	@rm -rf objs workspace/exec

debug :
	@echo $(I)

.PHONY : run clean debug

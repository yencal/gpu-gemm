NVCC = nvcc
ARCH = sm_80
CFLAGS = -O3 -arch=$(ARCH) --std=c++17

TARGET = gemm_bench

$(TARGET): src/main.cu src/*.cuh
	$(NVCC) $(CFLAGS) -lcurand -lcublas -o $@ src/main.cu

profile: src/profile.cu src/*.cuh
	$(NVCC) $(CFLAGS) -lcurand -lcublas -o $@ src/profile.cu

clean:
	rm -f $(TARGET) profile *.csv

.PHONY: clean
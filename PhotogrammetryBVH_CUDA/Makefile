# Compiler
NVCC := nvcc

CURRENT_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

# Compiler flags
CFLAGS := -std=c++17 -I$(CURRENT_DIR) $(shell pkg-config --cflags opencv4)

# CUDA flags release mode and debug mode
CUDAFLAGS_DEBUG := -arch=sm_86 -G -g

CUDAFLAGS_RELEASE := -arch=sm_86 -O3

# Source files
SRCS := area.cu

# Object files
OBJS := $(SRCS:.cu=.o)

# Target executable
TARGET := area

# Default target
all: CFLAGS += $(CUDAFLAGS_RELEASE)
all: $(TARGET)

# Debug target
debug: CFLAGS += $(CUDAFLAGS_DEBUG)
debug: $(TARGET)


# Rule to compile CUDA code
%.o: %.cu 
	$(NVCC) $(CFLAGS) -c $< -o $@

# Rule to link object files
$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) $(shell pkg-config --libs opencv4) -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)

# Run rule
run:
	./$(TARGET) ../MeshroomCache/Texturing/045fe3d998fc06f0cda1d7e59dd0b439f20888c0/texturedMesh.obj ../MeshroomCache/StructureFromMotion/e1ef6854b72357d92c404d2704058958e73c4aa4/cameras.sfm 201681465 ../figure/2DMask.jpg ../figure/cropped.obj

# build raycast

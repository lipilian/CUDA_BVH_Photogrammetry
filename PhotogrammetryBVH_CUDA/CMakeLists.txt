cmake_minimum_required(VERSION 3.10)

# Project name and C++ standard
project(MaskTrace LANGUAGES CXX CUDA)
#include_directories(${CMAKE_SOURCE_DIR}/include)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(Torch_DIR "/libtorch/share/cmake/Torch")

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)
# Set CUDA flags for release mode and debug mode.
set(CUDA_ARCHITECTURES 89)


set(SOURCES1 
    src/main.cpp 
    src/LiuHongCudaRayTracing.cu    
)

set(SOURCES2
    src/predict.cpp
    src/LiuHongSKINMASK.cpp
    src/LiuHongStereo.cpp
)

set(SOURCES3
    src/burnAI.cpp 
    src/LiuHongSKINMASK.cpp
    src/LiuHongAIMASK.cpp
)


add_executable(${PROJECT_NAME} ${SOURCES1})
add_executable(prepData ${SOURCES2})
add_executable(burnAI ${SOURCES3})

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include ${OPTIX_SDK_PATH}/include)            
target_include_directories(prepData PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(burnAI PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${CUDA_LIBRARIES})    
target_link_libraries(prepData ${OpenCV_LIBS} OpenMP::OpenMP_CXX ${TORCH_LIBRARIES})
target_link_libraries(burnAI ${OpenCV_LIBS} OpenMP::OpenMP_CXX ${TORCH_LIBRARIES})

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:DEBUG>:-g ${CUDA_COMMON_FLAGS}>
    $<$<CONFIG:RELEASE>:-O3 ${CUDA_COMMON_FLAGS}>
)

target_compile_options(burnAI PRIVATE
    $<$<CONFIG:DEBUG>:-g>
    $<$<CONFIG:RELEASE>:-O3>
)

target_compile_options(prepData PRIVATE
    $<$<CONFIG:DEBUG>:-g>
    $<$<CONFIG:RELEASE>:-O3>
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:DEBUG>:-arch=sm_${CUDA_ARCHITECTURES} -g -G>>
    $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:RELEASE>:-arch=sm_${CUDA_ARCHITECTURES} -O3>>
)

add_custom_target(remove
    COMMAND rm -rf ./*
    COMMENT "Cleaning all generated files"
)

add_custom_target(run 
    COMMAND ./${PROJECT_NAME} /workdir/TBSAcuda/TBSA_configure.json
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)







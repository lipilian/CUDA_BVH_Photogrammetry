# Purpose of this repository
- This Repo I want to demonstrate how to use photogrammetry and cuda projection projects.

1. Utilize photogrammetry to create 3D obj mesh with color (Meshroom)

2. Once you have the 3D obj mesh, we can project any image mask from 2D pixel (scanned image user used for photogrammetry) to 3D mesh (obj file) by using BVH tree data structure for fast raycasting. 

3. Mesh voting mechanism is constructed inside GPU with CUDA streaming to further smooth the projection performance.

4. Unit convertion is conducted based on crestereo (you can also use Lidar), which calibrate each pixel to 3D mesh distance to real world units.

# 0. Environment setup
Use Dockerfile and NVIDIA Container Toolkit to build the docker environment. 
# 1. Photogrammetry
Meshroom take care of the photogrammetry process. 
- Input: 
    - N image scans
    - N masks (optional)
- Output: 
    - cameras.sfm (camera intrinsic and extrinsic parameters)
    - mesh.obj (3D mesh)
    - texture.jpg (color texture)
    - color wrapping information (uv map)


cmake_minimum_required(VERSION 3.12)
project(lineMatch)

set(CMAKE_CXX_STANDARD 11)

# 寻找OpenCV库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})

# 包含include文件夹
include_directories(include)

# 添加头文件
set(DIR_Head
        include/opencv2/line_descriptor/descriptor.hpp
        include/opencv2/line_descriptor.hpp
        include/opencv2/lsd.hpp
        )

# 添加代码文件
set(DIR_Src
        src/binary_descriptor.cpp
        src/binary_descriptor_matcher.cpp
        src/bitarray.hpp
        src/bitops.hpp
        src/draw.cpp
        src/lsd.cpp
        src/LSDDetector.cpp
        src/precomp.hpp
        src/types.hpp
        examples/linematch.cpp)

# 生成LineDescriptor静态库用于调用
add_library(LineDescriptor SHARED ${DIR_Head} ${DIR_Src})
target_link_libraries(LineDescriptor
        opencv_features2d
        opencv_imgproc
        opencv_highgui)

# 生成可执行文件
# 基础演示和使用参考LineMatch，其余为更详细的演示
add_executable(BasicLineMatch examples/linematch.cpp)
target_link_libraries(BasicLineMatch LineDescriptor)

add_executable(ComputeDescriptors examples/others/compute_descriptors.cpp)
target_link_libraries(ComputeDescriptors LineDescriptor)

add_executable(KNNMatching examples/others/knn_matching.cpp)
target_link_libraries(KNNMatching LineDescriptor)

add_executable(LinesExtraction examples/others/lines_extraction.cpp)
target_link_libraries(LinesExtraction LineDescriptor)

add_executable(LSDLinesExtration examples/others/lsd_lines_extraction.cpp)
target_link_libraries(LSDLinesExtration LineDescriptor)

add_executable(RadiusMatching examples/others/radius_matching.cpp)
target_link_libraries(RadiusMatching LineDescriptor)

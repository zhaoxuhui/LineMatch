# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /root/programs/clion-2018.2/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /root/programs/clion-2018.2/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/CLionProjects/lineMatch

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/CLionProjects/lineMatch/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/testLineMatch.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/testLineMatch.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testLineMatch.dir/flags.make

CMakeFiles/testLineMatch.dir/examples/linematch.cpp.o: CMakeFiles/testLineMatch.dir/flags.make
CMakeFiles/testLineMatch.dir/examples/linematch.cpp.o: ../examples/linematch.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/CLionProjects/lineMatch/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testLineMatch.dir/examples/linematch.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/testLineMatch.dir/examples/linematch.cpp.o -c /root/CLionProjects/lineMatch/examples/linematch.cpp

CMakeFiles/testLineMatch.dir/examples/linematch.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testLineMatch.dir/examples/linematch.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /root/CLionProjects/lineMatch/examples/linematch.cpp > CMakeFiles/testLineMatch.dir/examples/linematch.cpp.i

CMakeFiles/testLineMatch.dir/examples/linematch.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testLineMatch.dir/examples/linematch.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /root/CLionProjects/lineMatch/examples/linematch.cpp -o CMakeFiles/testLineMatch.dir/examples/linematch.cpp.s

# Object files for target testLineMatch
testLineMatch_OBJECTS = \
"CMakeFiles/testLineMatch.dir/examples/linematch.cpp.o"

# External object files for target testLineMatch
testLineMatch_EXTERNAL_OBJECTS =

testLineMatch: CMakeFiles/testLineMatch.dir/examples/linematch.cpp.o
testLineMatch: CMakeFiles/testLineMatch.dir/build.make
testLineMatch: libLineDescriptor.so
testLineMatch: /usr/local/lib/libopencv_features2d.so.3.4.0
testLineMatch: /usr/local/lib/libopencv_flann.so.3.4.0
testLineMatch: /usr/local/lib/libopencv_highgui.so.3.4.0
testLineMatch: /usr/local/lib/libopencv_videoio.so.3.4.0
testLineMatch: /usr/local/lib/libopencv_imgcodecs.so.3.4.0
testLineMatch: /usr/local/lib/libopencv_imgproc.so.3.4.0
testLineMatch: /usr/local/lib/libopencv_core.so.3.4.0
testLineMatch: /usr/local/lib/libopencv_cudev.so.3.4.0
testLineMatch: CMakeFiles/testLineMatch.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/CLionProjects/lineMatch/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testLineMatch"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testLineMatch.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testLineMatch.dir/build: testLineMatch

.PHONY : CMakeFiles/testLineMatch.dir/build

CMakeFiles/testLineMatch.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testLineMatch.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testLineMatch.dir/clean

CMakeFiles/testLineMatch.dir/depend:
	cd /root/CLionProjects/lineMatch/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/CLionProjects/lineMatch /root/CLionProjects/lineMatch /root/CLionProjects/lineMatch/cmake-build-debug /root/CLionProjects/lineMatch/cmake-build-debug /root/CLionProjects/lineMatch/cmake-build-debug/CMakeFiles/testLineMatch.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testLineMatch.dir/depend


# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/matheus/Algoritmos/Vision/CPP/CatEM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/matheus/Algoritmos/Vision/CPP/CatEM/build

# Include any dependencies generated for this target.
include CMakeFiles/CatEM.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CatEM.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CatEM.dir/flags.make

CMakeFiles/CatEM.dir/CatEM.cpp.o: CMakeFiles/CatEM.dir/flags.make
CMakeFiles/CatEM.dir/CatEM.cpp.o: ../CatEM.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/matheus/Algoritmos/Vision/CPP/CatEM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CatEM.dir/CatEM.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CatEM.dir/CatEM.cpp.o -c /home/matheus/Algoritmos/Vision/CPP/CatEM/CatEM.cpp

CMakeFiles/CatEM.dir/CatEM.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CatEM.dir/CatEM.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/matheus/Algoritmos/Vision/CPP/CatEM/CatEM.cpp > CMakeFiles/CatEM.dir/CatEM.cpp.i

CMakeFiles/CatEM.dir/CatEM.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CatEM.dir/CatEM.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/matheus/Algoritmos/Vision/CPP/CatEM/CatEM.cpp -o CMakeFiles/CatEM.dir/CatEM.cpp.s

CMakeFiles/CatEM.dir/CatEM.cpp.o.requires:

.PHONY : CMakeFiles/CatEM.dir/CatEM.cpp.o.requires

CMakeFiles/CatEM.dir/CatEM.cpp.o.provides: CMakeFiles/CatEM.dir/CatEM.cpp.o.requires
	$(MAKE) -f CMakeFiles/CatEM.dir/build.make CMakeFiles/CatEM.dir/CatEM.cpp.o.provides.build
.PHONY : CMakeFiles/CatEM.dir/CatEM.cpp.o.provides

CMakeFiles/CatEM.dir/CatEM.cpp.o.provides.build: CMakeFiles/CatEM.dir/CatEM.cpp.o


# Object files for target CatEM
CatEM_OBJECTS = \
"CMakeFiles/CatEM.dir/CatEM.cpp.o"

# External object files for target CatEM
CatEM_EXTERNAL_OBJECTS =

CatEM: CMakeFiles/CatEM.dir/CatEM.cpp.o
CatEM: CMakeFiles/CatEM.dir/build.make
CatEM: /usr/local/lib/libopencv_viz.so.3.1.0
CatEM: /usr/local/lib/libopencv_videostab.so.3.1.0
CatEM: /usr/local/lib/libopencv_superres.so.3.1.0
CatEM: /usr/local/lib/libopencv_stitching.so.3.1.0
CatEM: /usr/local/lib/libopencv_shape.so.3.1.0
CatEM: /usr/local/lib/libopencv_photo.so.3.1.0
CatEM: /usr/local/lib/libopencv_objdetect.so.3.1.0
CatEM: /usr/local/lib/libopencv_calib3d.so.3.1.0
CatEM: /usr/local/lib/libopencv_features2d.so.3.1.0
CatEM: /usr/local/lib/libopencv_ml.so.3.1.0
CatEM: /usr/local/lib/libopencv_highgui.so.3.1.0
CatEM: /usr/local/lib/libopencv_videoio.so.3.1.0
CatEM: /usr/local/lib/libopencv_imgcodecs.so.3.1.0
CatEM: /usr/local/lib/libopencv_flann.so.3.1.0
CatEM: /usr/local/lib/libopencv_video.so.3.1.0
CatEM: /usr/local/lib/libopencv_imgproc.so.3.1.0
CatEM: /usr/local/lib/libopencv_core.so.3.1.0
CatEM: CMakeFiles/CatEM.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/matheus/Algoritmos/Vision/CPP/CatEM/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable CatEM"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CatEM.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CatEM.dir/build: CatEM

.PHONY : CMakeFiles/CatEM.dir/build

CMakeFiles/CatEM.dir/requires: CMakeFiles/CatEM.dir/CatEM.cpp.o.requires

.PHONY : CMakeFiles/CatEM.dir/requires

CMakeFiles/CatEM.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CatEM.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CatEM.dir/clean

CMakeFiles/CatEM.dir/depend:
	cd /home/matheus/Algoritmos/Vision/CPP/CatEM/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/matheus/Algoritmos/Vision/CPP/CatEM /home/matheus/Algoritmos/Vision/CPP/CatEM /home/matheus/Algoritmos/Vision/CPP/CatEM/build /home/matheus/Algoritmos/Vision/CPP/CatEM/build /home/matheus/Algoritmos/Vision/CPP/CatEM/build/CMakeFiles/CatEM.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CatEM.dir/depend


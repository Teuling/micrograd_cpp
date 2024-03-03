# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("microgradc++")

# Add the include directory
include_directories("C:/Users/Sieme/AppData/Local/armadillo-12.8.0/include")
include_directories("C:/Users/Sieme/source/repos/Graphviz-10.0.1/include/")

# Include sub-projects.
add_subdirectory ("microgradc++")

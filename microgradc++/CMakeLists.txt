# CMakeList.txt : CMake project for microgradc++, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

link_directories(C:/Users/Sieme/source/repos/Graphviz-10.0.1/lib)

# Add source to this project's executable.
add_executable (microgradc++ "microgradc++.cpp" "microgradc++.hpp" "engine.hpp" "sklearn.hpp" "tests.hpp")

target_link_libraries(microgradc++ cgraph cdt gvc)

# Copy files to the build directory using add_custom_command
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/plot.txt
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}//microgradc++/plot.txt
        ${CMAKE_CURRENT_BINARY_DIR}/plot.txt
    DEPENDS ${CMAKE_SOURCE_DIR}/microgradc++/plot.txt
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/data.txt
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}//microgradc++/data.txt
        ${CMAKE_CURRENT_BINARY_DIR}/data.txt
    DEPENDS ${CMAKE_SOURCE_DIR}/microgradc++/data.txt
)

# Create a custom target to ensure the file is copied during the build
add_custom_target(
    CopyFiles
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/plot.txt
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/data.txt
)

# Add dependency on the custom target to ensure the file is copied before building
add_dependencies(microgradc++ CopyFiles)


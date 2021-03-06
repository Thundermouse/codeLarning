project("developLearning" LANGUAGE CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE)DIR}/cmake/")

add_subdirectory(src)
add_subdirectory(samples)
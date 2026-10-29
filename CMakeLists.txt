cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.1/bin/nvcc)
set(CMAKE_EXPORT_COMPILE_COMMANDS true)

project(finalProj LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_executable(${PROJECT_NAME}
    src/raytracer.cu
    src/bvh.cu
    src/Error.cpp
    src/Utils.cpp
    src/main.cpp
)

add_executable(particle_sim
    src/raytracer.cu
    src/bvh.cu
    src/particles_calculation.cu
    src/Error.cpp
    src/Utils.cpp
    src/particles_sim.cpp
)

target_sources(${PROJECT_NAME} PRIVATE dep/glad/src/glad.c)
target_sources(particle_sim PRIVATE dep/glad/src/glad.c)

add_subdirectory(dep/glfw)

find_package(CUDAToolkit REQUIRED)
find_package(glm REQUIRED)
find_package(OpenMP)

# Link all necessary librairies
target_link_libraries(${PROJECT_NAME} PRIVATE
    CUDA::cudart        # Shared runtime
    CUDA::cuda_driver   # Driver API
    glfw
    glm::glm
    OpenMP::OpenMP_CXX # Not used but you never know ^^
)

target_link_libraries(particle_sim PRIVATE
    CUDA::cudart        # Shared runtime
    CUDA::cuda_driver   # Driver API
    glfw
    glm::glm
    OpenMP::OpenMP_CXX # Not used but you never know ^^
)


target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda
        --use_fast_math
        -O3
    >
)

target_compile_options(particle_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda
        --use_fast_math
        -O3
    >
)


set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

set_target_properties(particle_sim PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

add_custom_command(TARGET ${PROJECT_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}> ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_command(TARGET particle_sim
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:particle_sim> ${CMAKE_CURRENT_SOURCE_DIR}
)

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS} 
    "dep/glad/include/"
    "include"
)

target_include_directories(particle_sim PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS} 
    "dep/glad/include/"
    "include"
)

set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(particle_sim PROPERTIES LINKER_LANGUAGE CXX)

# I had a horrendous bug (3-4 hours of debugging) because of a shadow declaration, so I get cautious
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wshadow)
endif()



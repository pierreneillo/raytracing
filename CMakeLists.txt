cmake_minimum_required(VERSION 3.5)

project(finalProj LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenGL_GL_PREFERENCE "GLVND")

find_package(OpenGL REQUIRED)

find_package(glm CONFIG REQUIRED)

add_library(glad STATIC dep/glad/src/glad.c)

target_include_directories(glad PUBLIC dep/glad/include)

add_executable(finalProj
    src/main.cpp
    src/Error.cpp
    src/Mesh.cpp
    src/ShaderProgram.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    glad
    OpenGL::GL
    glfw
    glm::glm
    ${CMAKE_DL_LIBS}
)

add_custom_command(TARGET finalProj POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    $<TARGET_FILE:finalProj>
    ${CMAKE_CURRENT_SOURCE_DIR}
)

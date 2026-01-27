cmake_minimum_required(VERSION 3.5)

SET(CMAKE_EXPORT_COMPILE_COMMANDS 1)
SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_COMPILER g++)

project(finalProj)

add_executable(
  finalProj
  src/main.cpp
  src/Error.cpp
  src/Mesh.cpp
  src/ShaderProgram.cpp)


add_subdirectory(dep/glad)
target_link_libraries(${PROJECT_NAME} PRIVATE glad)

target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

if(NOT EXISTS "dep/glm")
  find_package(glm REQUIRED)
  if(GLM_FOUND)
    include_directories("${GLM_INCLUDE_DIR}")
  endif()
else()
  add_subdirectory(dep/glm)
  target_link_libraries(${PROJECT_NAME} PRIVATE glm)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})

add_custom_command(TARGET finalProj
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:finalProj> ${CMAKE_CURRENT_SOURCE_DIR})

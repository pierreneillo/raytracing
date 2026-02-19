

#include "Camera.hpp"
#include "Error.hpp"
#include "Utils.hpp"
#include <cstddef>
#include <cstdio>
#include <driver_types.h>
#include <glm/fwd.hpp>
#include <glm/matrix.hpp>
#include <memory>
#include <ostream>

#define fov 45.f

Camera *camPtr = nullptr;
ind particleCount;

cudaGraphicsResource *current_pos_buffer;

// *****************
// *  Shader mgmt  *
// *****************

GLuint compileShader(GLenum type, const std::string src) {
  // std::cout << "Compiling from shader source '" << src << "'" << std::endl;
  GLuint s = glCreateShader(type);

  std::ifstream input(src.c_str());
  if (!input)
    throw std::ios_base::failure("Cannot open source");

  std::stringstream buffer;
  buffer << input.rdbuf();
  std::string src_string = buffer.str();
  if (src_string.empty()) {
    std::cerr << "Source is empty" << std::endl;
    glDeleteShader(s);
    return -1;
  }
  const GLchar *src_string_gl = (const GLchar *)src_string.c_str();
  glShaderSource(s, 1, &src_string_gl, nullptr);
  glCompileShader(s);

  GLint ok;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[1024];
    glGetShaderInfoLog(s, 1024, nullptr, log);
    printf("Shader error:\n%s\n", log);
  }
  return s;
}

GLuint createBasicProgram() {
  GLuint program = glCreateProgram();
  GLuint vs = compileShader(GL_VERTEX_SHADER, "src/circle.vert");
  GLuint fs = compileShader(GL_FRAGMENT_SHADER, "src/circle.frag");

  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);

  glDeleteShader(vs);
  glDeleteShader(fs);
  return program;
}

// ***************
// *  Callbacks  *
// ***************

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);

  aspect = (float)width / (float)height;
}

void printHelp() {
  std::cout << "Help : " << std::endl
            << "\tEsc, W\tQuit the application" << std::endl;
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                 int mods) {
  if (action == GLFW_PRESS && key == GLFW_KEY_H) {
    printHelp();
  } else if ((action == GLFW_PRESS && key == GLFW_KEY_A) ||
             (action == GLFW_PRESS && key == GLFW_KEY_ESCAPE)) {
    glfwSetWindowShouldClose(window, true);
  } else if (action == GLFW_PRESS && key == GLFW_KEY_P) {
    // Map the current draw buffer to CPU memory
    gpuErrchk(cudaGraphicsMapResources(1, &current_pos_buffer, 0));
    real3 *d_pos_read;
    gpuErrchk(cudaGraphicsResourceGetMappedPointer(
        (void **)&d_pos_read, nullptr, current_pos_buffer));

    // Copy to CPU array
    real3 *h_pos = new real3[particleCount];
    gpuErrchk(cudaMemcpy(h_pos, d_pos_read, sizeof(real3) * particleCount,
                         cudaMemcpyDeviceToHost));

    // Save to file
    FILE *f = fopen("particles.bin", "wb");
    fwrite(h_pos + nStatic, sizeof(real3), particleCount - nStatic, f);
    fclose(f);

    // Cleanup
    delete[] h_pos;
    gpuErrchk(cudaGraphicsUnmapResources(1, &current_pos_buffer, 0));
    std::cout << "Saved " << particleCount << " particles to particles.bin"
              << std::endl;
  } else if (action == GLFW_PRESS && key == GLFW_KEY_SPACE) {
    // We start / stop the sim
    playing = !playing;
  }
}

void cursorPositionCallback(GLFWwindow *window, double posx, double posy) {

  // We need to find the position along the cursor ray that matches z = 0
  int w, h;
  glfwGetFramebufferSize(window, &w, &h);
  float nx = 2. * posx / (float)w - 1.;
  float ny = 1 - 2. * posy / (float)h;

  // We get two points along the ray
  mat4 invMat = glm::inverse(camPtr->viewProjMat());
  vec4 a4 = invMat * vec4(nx, ny, -1, 1);
  vec4 b4 = invMat * vec4(nx, ny, 1, 1);
  vec3 a = vec3(a4) / a4.w;
  vec3 b = vec3(b4) / b4.w;

  // We parametrize the ray as r = a + t (b - a)
  // We solve for z = resZ / 2 : t = - az / (bz - az)
  float t = ((getResZ() / 2) - a.z) / (b.z - a.z);
  vec3 simPos = a + t * (b - a);

  setCursorPos(real2({simPos.x, simPos.y}));

  if (moving) {
    if (camPtr)
      camPtr->rotate(vec2({posx, posy}) - moveOrigin);
    moveOrigin = {posx, posy};
  }
}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    if (action == GLFW_PRESS) {
      setCursorMagnitude(MAGNITUDE);
    } else if (action == GLFW_RELEASE) {
      setCursorMagnitude(0);
    }
  } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    if (action == GLFW_PRESS) {
      setCursorMagnitude(-2 * MAGNITUDE);
    } else if (action == GLFW_RELEASE) {
      setCursorMagnitude(0);
    }
  } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
    if (action == GLFW_PRESS) {
      moving = true;
      double xpos, ypos;
      glfwGetCursorPos(window, &xpos, &ypos);
      moveOrigin = {xpos, ypos};
    } else if (action == GLFW_RELEASE) {
      moving = false;
    }
  }
}

void scrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
  if (camPtr)
    camPtr->zoomin(yoffset);
}

/*Init OpenGL, GLFW and glad*/
int initGLX() {
  // Initialize GLFW
  if (!glfwInit())
    return -1;

  // Setup window with parameters
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Setup debug

  // Create window
  window = glfwCreateWindow(1920, 1080, "Particle simulation", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetKeyCallback(window, keyCallback);
  glfwSetCursorPosCallback(window, cursorPositionCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);
  glfwSetScrollCallback(window, scrollCallback);

  // Initialize GLAD
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }
  glDebugMessageCallback(debugMessageCallback, 0);
  return 0;
}

real3 *initParticles() {
  // Memory allocation
  ind posBufferSize = 1000;
  real3 *h_pos = (real3 *)malloc(sizeof(real3) * posBufferSize);

  // Borders
  // Left and right walls
  fillRectangle(0, 0, 0, 2. * s.h, getResY(), getResZ(), h_pos, nParticles,
                posBufferSize, s.h);
  fillRectangle(getResX() - 2. * s.h, 0, 0, getResX(), getResY(), getResZ(),
                h_pos, nParticles, posBufferSize, s.h);

  // Bottom and top walls
  fillRectangle(2. * s.h, 0, 0, getResX() - 2. * s.h, 2. * s.h, getResZ(),
                h_pos, nParticles, posBufferSize, s.h);
  fillRectangle(2. * s.h, getResY() - 2. * s.h, 0, getResX() - 2. * s.h,
                getResY(), getResZ(), h_pos, nParticles, posBufferSize, s.h);

  // Back and front walls
  fillRectangle(2. * s.h, 2. * s.h, 0, getResX() - 2. * s.h,
                getResY() - 2. * s.h, 2. * s.h, h_pos, nParticles,
                posBufferSize, s.h);
  fillRectangle(2. * s.h, 2. * s.h, getResZ() - 2. * s.h, getResX() - 2. * s.h,
                getResY() - 2. * s.h, getResZ(), h_pos, nParticles,
                posBufferSize, s.h);

  printf("%d static particles\n", nParticles);
  nStatic = nParticles;

  // Particles
  // We fill in the standford bunny
  TriangleMesh tm = loadMesh("data/monk.obj");
  // We transform it to fit the bounds fo the simulation (padded)
  AABB simBB = {
      real3(8 * s.h, 8 * s.h, 8 * s.h),
      real3(getResX() - 8 * s.h, getResY() - 8 * s.h, getResZ() - 8 * s.h)};
  AABB meshBB =
      AABB({real3(10e30, 10e30, 10e30), real3(-10e30, -10e30, -10e30)});
  for (int i = 0; i < tm.nPrimitives; i++) {
    tm.add_to_bb(i, meshBB);
  }
  // We aim to transform to fit, so first the extent must fit :
  real3 simExtent = simBB.M - simBB.m;
  real3 meshExtent = meshBB.M - meshBB.m;

  // Extent ratio, we have to scale by the min of the ratios in each direction,
  // to avoid deformation
  real3 eR = simExtent / meshExtent;

  real scale = fminf(eR.x, fminf(eR.y, eR.z));

  // Secondly the translation
  real3 simCenter = simBB.m + simExtent / 2.;
  real3 meshCenter = meshBB.m + meshExtent / 2.;

  real3 translation = simCenter - (meshCenter * scale);

  std::cout << "Transformation of scale " << scale << "and of translation"
            << translation.x << "," << translation.y << "," << translation.z
            << std::endl;

  // We then apply the transform
  // We dont touch the normals as there is no rotation
  for (int i = 0; i < tm.nVerts; i++) {
    tm.vertices[i] = tm.vertices[i] * scale + translation;
  }

  Object bunny = Object({.type = UNDEF_MATERIAL}, tm);

  fillMesh(bunny, h_pos, nParticles, posBufferSize, s.h);

  delete[] tm.vertices;
  delete[] tm.normals;
  delete[] tm.triangles;

  //  fillRectangle(1, 1, 1, 17, 17, 17, h_pos, nParticles, posBufferSize, .5);

  printf("%d total particles\n", nParticles);

  return h_pos;
}

int main() {
  setenv("__NV_PRIME_RENDER_OFFLOAD", "1", 1);
  setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia", 1);
  if (initGLX()) {
    return -1;
  };

  // Compile Shaders and create program
  program = createBasicProgram();

  // Get particle positions
  real3 *positions = initParticles();
  initSimulation(positions, false);
  particleCount = nParticles;
  std::cout << "Total particle count : " << particleCount << std::endl;
  // We create the SSBO and an associated CUDA Graphics ressource
  // We create a list of N_BUFFERS buffers, that will circle, to prevent
  // synchronous calculations and rendering, which results in a ping pong that
  // slows the performances.
  GLuint pos_ssbo[N_BUFFERS];
  cudaGraphicsResource *cuda_pos_res[N_BUFFERS];
  glCreateBuffers(N_BUFFERS, pos_ssbo);
  GLuint vel_ssbo[N_BUFFERS];
  cudaGraphicsResource *cuda_vel_res[N_BUFFERS];
  glCreateBuffers(N_BUFFERS, vel_ssbo);

  for (int i = 0; i < N_BUFFERS; i++) {
    glNamedBufferStorage(pos_ssbo[i], sizeof(real3) * particleCount, positions,
                         GL_DYNAMIC_STORAGE_BIT);
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&cuda_pos_res[i], pos_ssbo[i],
                                           cudaGraphicsMapFlagsNone));
    glNamedBufferStorage(
        vel_ssbo[i], sizeof(real3) * particleCount,
        std::vector<real3>(particleCount, real3(0., 0., 0.)).data(),
        GL_DYNAMIC_STORAGE_BIT);
    gpuErrchk(cudaGraphicsGLRegisterBuffer(&cuda_vel_res[i], vel_ssbo[i],
                                           cudaGraphicsMapFlagsNone));
  }

  // TODO : correct this, might not work with more that 3 buffers, or depending
  // on the sync
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pos_ssbo[N_BUFFERS - 1]);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vel_ssbo[N_BUFFERS - 1]);

  float z_dist = getResY() / (2.0f * tanf(glm::radians(fov / 2.0f)));

  static Camera cam =
      Camera({getResX() / 2.0f, getResY(), -z_dist},
             {getResX() / 2.0f, getResY() / 2.0f, getResZ() / 2.0f});

  camPtr = &cam;

  GLuint vao;
  glCreateVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);

  int w, h;
  glfwGetFramebufferSize(window, &w, &h);
  aspect = (float)w / (float)h;

  GLint radiusLoc = glGetUniformLocation(program, "radius");
  GLint projLoc = glGetUniformLocation(program, "projection");
  GLint staticLoc = glGetUniformLocation(program, "nStatic");
  GLint passTypeLoc = glGetUniformLocation(program, "passType");

  glUseProgram(program);
  glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(cam.viewProjMat()));
  glUniform1f(radiusLoc, s.h / 2.);
  glUniform1i(staticLoc, nStatic);

  if (recording)
    startRecording(w, h);

  // Initialise position file
  FILE *f = fopen("positions.txt", "w");
  fprintf(f, "%d\n", particleCount - nStatic);

  float last_time = glfwGetTime() - (float)1. / TARGET_FPS;
  float current_time = glfwGetTime();
  while (!glfwWindowShouldClose(window)) {
    current_time = glfwGetTime();
    float delta_time = current_time - last_time;
    last_time = current_time;
    std::cout << (int)(1. / delta_time) << " FPS"
              << "     \r";
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(cam.viewProjMat()));
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, pos_ssbo[getDrawInd()]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, vel_ssbo[getDrawInd()]);

    // Two pass system to draw both transparent and opaque particles (static are
    // rendered transparent for debug)
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    glUniform1i(passTypeLoc, 0); // Firstly : the opaque pass

    glDrawArraysInstanced(GL_TRIANGLES, 0, 240, particleCount);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    glUniform1i(passTypeLoc,
                1); // Secondly : the transparent pass

    glDrawArraysInstanced(GL_TRIANGLES, 0, 240, particleCount);

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    glfwSwapBuffers(window);
    if (recording)
      recordFrame(w, h);
    glfwPollEvents();

    if (playing && cudaStreamQuery(stream) == cudaSuccess) {
      // Save positions buffer
      gpuErrchk(
          cudaGraphicsMapResources(1, &cuda_pos_res[getReadInd()], stream));
      real3 *d_pos_read_temp;
      gpuErrchk(cudaGraphicsResourceGetMappedPointer(
          (void **)&d_pos_read_temp, nullptr, cuda_pos_res[getReadInd()]));
      real3 *h_pos_temp = new real3[particleCount];
      gpuErrchk(cudaMemcpy(h_pos_temp, d_pos_read_temp,
                           sizeof(real3) * particleCount,
                           cudaMemcpyDeviceToHost));
      for (ind i = nStatic; i < particleCount; i++) {
        fprintf(f, "%f %f %f\n", h_pos_temp[i].x, h_pos_temp[i].y,
                h_pos_temp[i].z);
      }
      delete[] h_pos_temp;
      gpuErrchk(
          cudaGraphicsUnmapResources(1, &cuda_pos_res[getReadInd()], stream));
      incrementBufferInd();
      current_pos_buffer = cuda_pos_res[getDrawInd()];
      // The stream has finished running, we step
      // We lock both the read and the write ressource
      // TODO : merge into one call
      gpuErrchk(
          cudaGraphicsMapResources(1, &cuda_pos_res[getReadInd()], stream));
      gpuErrchk(
          cudaGraphicsMapResources(1, &cuda_pos_res[getWriteInd()], stream));
      gpuErrchk(
          cudaGraphicsMapResources(1, &cuda_vel_res[getReadInd()], stream));
      gpuErrchk(
          cudaGraphicsMapResources(1, &cuda_vel_res[getWriteInd()], stream));
      const real3 *d_pos_read, *d_vel_read;
      real3 *d_pos_write, *d_vel_write;
      gpuErrchk(cudaGraphicsResourceGetMappedPointer(
          (void **)&d_pos_read, nullptr, cuda_pos_res[getReadInd()]));
      gpuErrchk(cudaGraphicsResourceGetMappedPointer(
          (void **)&d_pos_write, nullptr, cuda_pos_res[getWriteInd()]));
      gpuErrchk(cudaGraphicsResourceGetMappedPointer(
          (void **)&d_vel_read, nullptr, cuda_vel_res[getReadInd()]));
      gpuErrchk(cudaGraphicsResourceGetMappedPointer(
          (void **)&d_vel_write, nullptr, cuda_vel_res[getWriteInd()]));

      // We compute the number of steps necessary for our physics
      // simulation

      stepSimulation(d_pos_read, d_pos_write, d_vel_read, d_vel_write,
                     min((ind)(1. / getdt() / TARGET_FPS),
                         (ind)100)); // TODO : change this

      gpuErrchk(
          cudaGraphicsUnmapResources(1, &cuda_pos_res[getReadInd()], stream));
      gpuErrchk(
          cudaGraphicsUnmapResources(1, &cuda_pos_res[getWriteInd()], stream));
      gpuErrchk(
          cudaGraphicsUnmapResources(1, &cuda_vel_res[getReadInd()], stream));
      gpuErrchk(
          cudaGraphicsUnmapResources(1, &cuda_vel_res[getWriteInd()], stream));
    }
  }

  endSimulation();
  fclose(f);
  if (recording)
    pclose(ffmpegPipe);
  for (int i = 0; i < N_BUFFERS; i++) {
    gpuErrchk(cudaGraphicsUnregisterResource(cuda_pos_res[i]));
    glDeleteBuffers(1, &pos_ssbo[i]);
  }
  glDeleteVertexArrays(1, &vao);
  glDeleteProgram(program);
  glfwTerminate();

  return 0;
}
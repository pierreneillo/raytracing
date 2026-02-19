#include "../include/utilities.cuh"

// *****************************
// *       INIT GLOBALS        *
// *****************************

extern cudaStream_t stream;

extern real2 cursorPos;
extern float cursorRadius;
extern float cursorMagnitude;

extern ind nParticles;
extern ind nStatic;

extern int scale;

// SPH coefficients
struct SPH {
  real nu = .2;           // viscosity coefficient
  real d0 = 700;          // rest density
  real h = .25;           // particle spacing (i.e., diameter)
  real hh = h / 2.;       // Half h
  real3 g = {0, -9.8, 0}; // gravity

  real eta = 0.01;
  real eh2 = eta * h * h;
  real gamma = 7.0;
  real c = abs(g.y) / eta;

  real m0 = d0 * cube(h);          // rest mass
  real k = d0 * square(c) / gamma; // EOS coefficient

  // simulation parameters
  real dt = 0.0001; // time step
  ind resX = 20 * scale;
  ind resY = 40 * scale;
  ind resZ = 20 * scale;
  real cellSize = 2. * h;
  real iCellSize = 1. / cellSize;
};

extern SPH s;

#define MAGNITUDE 100

float getdt();
uint getResX();
uint getResY();
uint getResZ();
void setCursorPos(real2 p);
void setCursorMagnitude(real m);

void fillRectangle(float startPosX, float startPosY, float startPosZ,
                   float endPosX, float endPosY, float endPosZ, real3 *&posptr,
                   ind &nParticles, ind &maxBufferSize, real h);

// Forward definition of the object class for fillMesh function's declaration
class Object;

void fillMesh(const Object &o, real3 *&pos, ind &nParticles, ind &maxBufferSize,
              float h);

real3 *initParticles();

void initSimulation(real3 *particlePositions, bool initPosNVel = true);
void stepSimulation(const real3 *d_pos_read, real3 *d_pos_write,
                    const real3 *d_vel_read, real3 *d_vel_write, ind n_steps);
void stepSimulation(ind n_steps);
const real3 *getSimulationDevicePositions();
void endSimulation();
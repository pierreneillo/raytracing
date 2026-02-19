#version 460 core

in vec3 localPos;
flat in int isStatic;
in vec3 baseColor;
out vec4 FragColor;

uniform float radius;
uniform int passType; // Two passes to draw transparent objects

void main() {
  if (passType == 0) {
    if (isStatic == 1) {
      discard;
    }
    FragColor = vec4(baseColor, 1);
  } else {
    if (isStatic == 0) {
      discard;
    }
    FragColor = vec4(baseColor, 0.05);
  }
}
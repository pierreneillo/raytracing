#version 450 core

in vec2 fTexCoord;

out vec4 FragColor;

layout(binding = 0) uniform sampler2D screenTexture;

void main() { FragColor = texture(screenTexture, fTexCoord); }
#version 450 core

in vec2 fTexCoord;

out vec4 colorOut;

uniform sampler2D screenTexture;

void main() { colorOut = texture(screenTexture, fTexCoord); }
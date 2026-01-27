#version 450 core            // minimal GL version support expected from the GPU

void main(){
  gl_FragDepth = gl_FragCoord.z;
}

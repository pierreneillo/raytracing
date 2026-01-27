#version 450 core // minimal GL version support expected from the GPU

struct LightSource {
  vec3 position;
  vec3 color;
  float intensity;
  int isActive;
};

int numberOfLights = 3;
uniform LightSource lightSources[3];
uniform mat4 lightSpaceMatrices[3];

struct Material {
  int type;
  bool uses_texture;
  sampler2D normalMap;
  sampler2D colorMap;
  vec3 ambientColor;
  vec3 diffuseColor;
  vec3 specularColor;
  float shininess;
  float kA;
  float kD;
  float kS;
};

uniform Material material;

uniform vec3 camPos;

in vec3 fPosition;
in vec3 fNormal;
in vec2 fTexCoord;

out vec4
    colorOut; // shader output: the color response attached to this fragment

float pi = 3.1415927;

void main() {
  vec3 _n = normalize(fNormal);
  vec3 v =
      normalize(camPos - fPosition); // unit view vector pointing to the camera
  vec3 ambient = material.ambientColor;
  vec3 n = _n;

  if (material.uses_texture) {
    // This is the value from the texture
    vec3 lN = vec3(texture(material.normalMap, fTexCoord)) * 2. - 1.;
    // Our aim is to reproduce the transformation (0,0,1) => `_n` and apply it
    // to `lN`, we build a tangent basis aligned with _n. We have to generate an
    // orthonormal basis for _n. We build the 'up' vector according to its z
    // value (to prevent problems with cross products later)
    vec3 up = (abs(_n.z) < 0.999) ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
    vec3 T = normalize(cross(up, _n)); // Tangent
    vec3 B = cross(_n, T);

    // We then apply the transform to the basis
    n = normalize(mat3(T, B, _n) * lN);

    vec3 baseColor = vec3(texture(material.colorMap, fTexCoord));
    ambient = baseColor;
  }

  vec3 diffuse = vec3(0, 0, 0);
  vec3 specular = vec3(0, 0, 0);
  for (int i = 0; i < numberOfLights; ++i) {
    LightSource light = lightSources[i];
    if (light.isActive == 1) { // consider active lights only
      vec3 li = normalize(light.position -
                          fPosition); // unit vector pointing to the light
      vec3 h = normalize(li + v);
      diffuse += max(dot(_n, li), 0) * light.color * material.diffuseColor;
      specular += pow(max(dot(_n, h), 0), material.shininess) * light.color *
                  material.specularColor;
    }
  }
  vec3 color =
      ambient * material.kA + diffuse * material.kD + specular * material.kS;
  colorOut = vec4(color, 1.0); // build an RGBA value from an RGB one
}

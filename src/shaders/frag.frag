#version 450
layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;
layout(binding = 0) uniform sampler2D texSampler;
void main() {
    if (inUV.x < 0.0 || inUV.x > 1.0 || inUV.y < 0.0 || inUV.y > 1.0) {
        discard;
    }
    outColor = texture(texSampler, inUV);
}
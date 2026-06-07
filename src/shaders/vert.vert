#version 450
layout(location = 0) out vec2 outUV;
layout(push_constant) uniform PushConst {
    vec2 offset;
    vec2 scale;
} pc;
void main() {
    vec2 pos[4] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0,  1.0),
        vec2( 1.0,  1.0)
    );
    gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
    outUV = (pos[gl_VertexIndex] * 0.5 + 0.5) * pc.scale + pc.offset;
}
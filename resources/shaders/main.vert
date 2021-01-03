#version 450

layout(location = 0) in vec2 in_vertex_position;
layout(location = 1) in vec2 in_vertex_uv;

layout(location = 0) out vec2 out_vertex_uv;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 uniform_transform;
};

void main() {
    gl_Position = uniform_transform * vec4(in_vertex_position, 0.0, 1.0);
    out_vertex_uv = vec2(in_vertex_uv.x, in_vertex_uv.y);
}
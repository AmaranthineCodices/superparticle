#version 450

layout(location = 0) in vec2 in_vertex_position;
layout(location = 1) in vec2 in_vertex_uv;
layout(location = 2) in vec2 in_size_mask;

layout(location = 0) out vec2 out_vertex_uv;
layout(location = 1) out vec4 out_instance_tint;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 uniform_transform;
};

struct Instance {
    vec2 pos;
    vec2 size;
    vec4 tint;
};

layout(set = 1, binding = 0) readonly buffer Instances {
    Instance instances[];
};

void main() {
    Instance instance_data = instances[gl_InstanceIndex];

    vec2 instanced_vertex_pos = instance_data.pos;
    instanced_vertex_pos += in_vertex_position;
    instanced_vertex_pos += instance_data.size * in_size_mask;

    gl_Position = uniform_transform * vec4(instanced_vertex_pos, 0.0, 1.0);
    out_vertex_uv = vec2(in_vertex_uv.x, in_vertex_uv.y);
    out_instance_tint = instance_data.tint;
}
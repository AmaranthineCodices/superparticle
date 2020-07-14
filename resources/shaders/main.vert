#version 450

layout(location = 0) in vec2 in_vertex_position;
layout(location = 1) in vec2 in_vertex_uv;
layout(location = 2) in vec2 in_size_mask;

layout(location = 0) out vec2 out_vertex_uv;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 uniform_transform;
};

layout(set = 1, binding = 0) buffer Instances {
    vec4 instances[];
};

void main() {
    vec4 instance_data = instances[gl_InstanceIndex];
    vec2 instance_pos = instance_data.xy;
    vec2 instance_size = instance_data.zw;

    vec2 instanced_vertex_pos = instance_pos;
    instanced_vertex_pos += in_vertex_position;
    instanced_vertex_pos += instance_size * in_size_mask;

    gl_Position = uniform_transform * vec4(instanced_vertex_pos, 0.0, 1.0);
    out_vertex_uv = vec2(in_vertex_uv.x, in_vertex_uv.y);
}
#version 450

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec4 in_tint;

layout(location = 0) out vec4 out_fragment_color;

layout(set = 1, binding = 1) uniform texture2D diffuse_texture;
layout(set = 1, binding = 2) uniform sampler diffuse_sampler;

void main() {
    out_fragment_color = texture(sampler2D(diffuse_texture, diffuse_sampler), in_uv) * in_tint;
}
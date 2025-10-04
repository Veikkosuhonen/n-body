
/**
Particle rendering
*/

struct Globals {
    view_matrix: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

const particle_size = 0.01;
const quad_offsets = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>( 1.0,  1.0)
);

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) frag_color: vec4<f32>,
};

@vertex
fn particle_vs_main(@location(0) in_pos: vec2<f32>, @location(1) in_vel: vec2<f32>, @builtin(vertex_index) vertex_id : u32) -> VertexOutput {
    let world_pos = in_pos + particle_size * quad_offsets[vertex_id];
    let position = globals.view_matrix * vec4<f32>(world_pos, 0.0, 1.0);

    var out: VertexOutput;
    out.position = position;
    let speed = length(in_vel) * 0.02;
    out.frag_color = vec4<f32>(1.0 - speed, 0.5 * (1.0 + speed), speed, 1.0);
    // out.frag_color = vec4<f32>(globals.view_matrix[1][0], globals.view_matrix[1][1], globals.view_matrix[1][2], 1.0);
    return out;
}

@fragment
fn particle_fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.frag_color * 0.2;
}

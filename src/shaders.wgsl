/*
N-body compute

define N and WORKGROUP_SIZE when compiling
*/

const N: u32 = <N>;
const WORKGROUP_SIZE: u32 = <WORKGROUP_SIZE>;

const COLLISION_DISTANCE: f32 = 0.03;
const MERGING_SPEED_THRESHOLD: f32 = 0.1;

struct Particle {
    pos: vec2<f32>,
    vel: vec2<f32>,
    // mass: f32,
    // _pad: f32,
}


@group(0) @binding(0) var<storage, read> input_particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> output_particles: array<Particle>;
/*
@compute @workgroup_size(WORKGROUP_SIZE)
fn n_body_merging_sim_main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i = global_id.x;
    if (i >= N) {
        return;
    }

    let current_particle = input_particles[i];

    if (current_particle.mass < 0.01) {
        return; // Skip massless particles
    }

    var force = vec2<f32>(0.0, 0.0);

    var new_mass = current_particle.mass;
    var new_vel = current_particle.vel;
    var new_pos = current_particle.pos;

    let dt = 0.00001;
    let softening = 0.01;

    for (var j = 0u; j < N; j = j + 1u) {
        if (j == i) {
            continue;
        }

        let other_particle = input_particles[j];

        if (other_particle.mass < 0.01) {
            continue; // Skip massless particles
        }

        let dp = other_particle.pos - current_particle.pos;
        let dist = length(dp);

        let relative_velocity = current_particle.vel - other_particle.vel;
        let relative_speed = length(relative_velocity);

        if (dist < COLLISION_DISTANCE && relative_speed < MERGING_SPEED_THRESHOLD) {
            // Merge particles
            // Smaller index absorbs the larger index. So if current idx is larger, it loses mass to 0.
            // This particle might still get lucky and gain mass from other particles in this same step.
            var mass_transfer = other_particle.mass;
            if (i > j) { // Loser
                mass_transfer = -current_particle.mass;
            }

            let total_mass = current_particle.mass + other_particle.mass;
            new_vel = (new_vel * current_particle.mass + other_particle.vel * other_particle.mass) / total_mass;
            new_pos = (new_pos * current_particle.mass + other_particle.pos * other_particle.mass) / total_mass;

            new_mass += mass_transfer;
        } else {
            let inv_dist = 1.0 / (dist + softening);
            let inv_dist_cubed = inv_dist * inv_dist * inv_dist;
            force += dp * inv_dist_cubed * other_particle.mass * current_particle.mass;
        }
    }

    new_vel += dt * force / current_particle.mass;
    new_pos += dt * new_vel;

    output_particles[i].pos = new_pos;
    output_particles[i].vel = new_vel;
    output_particles[i].mass = new_mass;
}
 */

@compute @workgroup_size(WORKGROUP_SIZE)
fn n_body_sim_main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let i = global_id.x;
    if (i >= N) {
        return;
    }

    let current_particle = input_particles[i];

    var force = vec2<f32>(0.0, 0.0);
    var new_vel = current_particle.vel;
    var new_pos = current_particle.pos;

    let dt = 0.00001;
    let softening = 0.01;

    for (var j = 0u; j < N; j = j + 1u) {
        if (j == i) {
            continue;
        }

        let other_particle = input_particles[j];

        let dp = other_particle.pos - current_particle.pos;
        let dist = length(dp);

        let inv_dist = 1.0 / (dist + softening);
        let inv_dist_cubed = inv_dist * inv_dist * inv_dist;
        force += dp * inv_dist_cubed;
    }

    new_vel += dt * force;
    new_pos += dt * new_vel;

    output_particles[i].pos = new_pos;
    output_particles[i].vel = new_vel;
}


/*
N-body compute with tiling optimization

define N, WORKGROUP_SIZE and TILE_SIZE when compiling
*/

const TILE_SIZE: u32 = <TILE_SIZE>;

var<workgroup> tile: array<Particle, <WORKGROUP_SIZE>>;

@compute @workgroup_size(<WORKGROUP_SIZE>)
fn n_body_sim_tiled_main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let current_particle = input_particles[global_id.x];
    var force = vec2<f32>(0.0, 0.0);

    let dt = 0.00001;
    let softening = 0.01;

    for (var tile_start = 0u; tile_start < N; tile_start += TILE_SIZE) {
        // Load tile into shared memory
        let idx = tile_start + local_id.x;
        if (idx < N) {
            tile[local_id.x] = input_particles[idx];
        }
        workgroupBarrier();

        // Interact with particles in tile
        let tile_size = min(TILE_SIZE, N - tile_start);

        for (var j = 0u; j < tile_size; j = j + 1u) {
            if (tile_start + j == global_id.x) {
                continue;
            }

            let other_particle = tile[j];

            let dp = other_particle.pos - current_particle.pos;
            let dist_sq = dot(dp, dp);
            let inv_dist = 1.0 / sqrt(dist_sq + softening);
            let inv_dist_cubed = inv_dist * inv_dist * inv_dist;
            force += dp * inv_dist_cubed;
        }

        // workgroupBarrier();
    }

    var new_vel = current_particle.vel + dt * force;
    var new_pos = current_particle.pos + dt * new_vel;

    output_particles[global_id.x].pos = new_pos;
    output_particles[global_id.x].vel = new_vel;
}

/**
Particle rendering
*/

const particle_size = 0.003;
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
    let size = particle_size;
    let position = vec4<f32>(in_pos + size * quad_offsets[vertex_id], 0.0, 1.0);
    var out: VertexOutput;
    out.position = position;
    let speed = length(in_vel) * 0.002;
    out.frag_color = vec4<f32>(1.0 - speed, 0.5 * (1.0 + speed), speed, 1.0);
    return out;
}

@fragment
fn particle_fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.frag_color * 0.1;
}

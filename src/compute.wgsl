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
}


@group(0) @binding(0) var<storage, read> input_particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> output_particles: array<Particle>;

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

    let dt = 0.0002;
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

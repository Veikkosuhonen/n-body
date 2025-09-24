(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const n of document.querySelectorAll('link[rel="modulepreload"]'))a(n);new MutationObserver(n=>{for(const t of n)if(t.type==="childList")for(const p of t.addedNodes)p.tagName==="LINK"&&p.rel==="modulepreload"&&a(p)}).observe(document,{childList:!0,subtree:!0});function s(n){const t={};return n.integrity&&(t.integrity=n.integrity),n.referrerPolicy&&(t.referrerPolicy=n.referrerPolicy),n.crossOrigin==="use-credentials"?t.credentials="include":n.crossOrigin==="anonymous"?t.credentials="omit":t.credentials="same-origin",t}function a(n){if(n.ep)return;n.ep=!0;const t=s(n);fetch(n.href,t)}})();const C=`/*
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

const particle_size = 0.002;
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
`,q=(u,e)=>{let s=u;return Object.keys(e).forEach(a=>{s=s.replaceAll(`<${a}>`,String(e[a]))}),s},R=document.querySelector("canvas"),D=document.querySelector("#timing-info"),l={compute:[],render:[]};async function W(){if(!navigator.gpu)throw new Error("WebGPU not supported on this browser.");const u=await navigator.gpu.requestAdapter();if(!u)throw new Error("No appropriate GPUAdapter found.");const e=await u.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:{}}),s=R.getContext("webgpu"),a=navigator.gpu.getPreferredCanvasFormat();s.configure({device:e,format:a,alphaMode:"premultiplied"}),new ResizeObserver(r=>{for(const i of r){const o=i.target,c=i.contentBoxSize[0].inlineSize,v=i.contentBoxSize[0].blockSize;o.width=Math.max(1,Math.min(c,e.limits.maxTextureDimension2D)),o.height=Math.max(1,Math.min(v,e.limits.maxTextureDimension2D))}}).observe(R);const t=5e4,p=256,E=q(C,{N:t,WORKGROUP_SIZE:p,TILE_SIZE:p}),_=4,d=new Float32Array(t*_);for(let r=0;r<t;++r){const i=2*Math.PI*Math.random(),o=.1+.7*Math.random(),c=o*Math.cos(i),v=o*Math.sin(i),h=220*o+10,b=-h*Math.sin(i),S=h*Math.cos(i);d[_*r+0]=c,d[_*r+1]=v,d[_*r+2]=b,d[_*r+3]=S}const g=e.createBuffer({label:"Particle buffer A",size:d.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(g,0,d);const P=e.createBuffer({label:"Particle buffer B",size:d.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),f=e.createQuerySet({type:"timestamp",count:4}),O=e.createBuffer({size:f.count*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),m=e.createBuffer({size:f.count*8,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),y=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),U=e.createBindGroup({layout:y,entries:[{binding:0,resource:{buffer:g}},{binding:1,resource:{buffer:P}}]}),B=e.createBindGroup({layout:y,entries:[{binding:0,resource:{buffer:P}},{binding:1,resource:{buffer:g}}]}),L=e.createPipelineLayout({bindGroupLayouts:[y]}),M=e.createShaderModule({label:"Compute shader",code:E}),T=e.createComputePipeline({layout:L,compute:{module:M,entryPoint:"n_body_sim_tiled_main"}}),x=e.createShaderModule({label:"Render shader",code:E}),N=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[]}),vertex:{module:x,entryPoint:"particle_vs_main",buffers:[{arrayStride:_*4,stepMode:"instance",attributes:[{shaderLocation:0,offset:0,format:"float32x2"},{shaderLocation:1,offset:8,format:"float32x2"}]}]},fragment:{module:x,entryPoint:"particle_fs_main",targets:[{format:a,blend:{color:{operation:"add",srcFactor:"one",dstFactor:"one"},alpha:{operation:"add",srcFactor:"one",dstFactor:"one"}}}]},primitive:{topology:"triangle-list"}});let w=0;async function G(){const r=e.createCommandEncoder(),i=r.beginComputePass({timestampWrites:{querySet:f,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}});i.setPipeline(T),w%2===0?i.setBindGroup(0,U):i.setBindGroup(0,B),i.dispatchWorkgroups(Math.ceil(t/p)),i.end();const o=r.beginRenderPass({colorAttachments:[{view:s.getCurrentTexture().createView(),loadOp:"clear",clearValue:{r:0,g:0,b:0,a:1},storeOp:"store"}],timestampWrites:{querySet:f,beginningOfPassWriteIndex:2,endOfPassWriteIndex:3}});o.setPipeline(N),w%2===0?o.setVertexBuffer(0,P):o.setVertexBuffer(0,g),o.draw(6,t),o.end(),r.resolveQuerySet(f,0,f.count,O,0),r.copyBufferToBuffer(O,0,m,0,m.size),e.queue.submit([r.finish()]),await m.mapAsync(GPUMapMode.READ);const c=new BigInt64Array(m.getMappedRange()),v=Number(c[1]-c[0])/1e6,h=Number(c[3]-c[2])/1e6;m.unmap(),l.compute.push(v),l.render.push(h),l.compute.length>10&&l.compute.shift(),l.render.length>10&&l.render.shift();const b=I=>I.reduce((z,j)=>z+j,0)/I.length,S=b(l.compute),A=b(l.render);D.textContent=`Compute: ${S.toFixed(3)}ms | Render: ${A.toFixed(3)}ms`,w++,requestAnimationFrame(G)}requestAnimationFrame(G)}W().catch(u=>{console.error(u)});

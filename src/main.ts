import "./style.css";

const canvasEl = document.querySelector("canvas")!;
const timingInfoEl = document.querySelector("#timing-info")!;

const timingsData = {
  compute: [] as number[],
  render: [] as number[],
};

async function main() {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
  }

  const device = await adapter.requestDevice({
    requiredFeatures: ["timestamp-query", "shader-f16"],
    requiredLimits: {
      maxComputeWorkgroupSizeX: 1024,
      maxComputeInvocationsPerWorkgroup: 1024,
    },
  });

  const context = canvasEl.getContext("webgpu")!;
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
  });

  const observer = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const canvas = entry.target as HTMLCanvasElement;
      const width = entry.contentBoxSize[0].inlineSize;
      const height = entry.contentBoxSize[0].blockSize;
      canvas.width = Math.max(
        1,
        Math.min(width, device.limits.maxTextureDimension2D),
      );
      canvas.height = Math.max(
        1,
        Math.min(height, device.limits.maxTextureDimension2D),
      );
    }
  });
  observer.observe(canvasEl);

  const numParticles = 60_000;
  const WORKGROUP_TILE_SIZE = 512;

  const structSize = 2 * 2; // floats

  const initialParticleData = new Float32Array(numParticles * structSize);
  for (let i = 0; i < numParticles; ++i) {
    const randomAngle1 = 2 * Math.PI * Math.random();
    const randomRadius = 0.1 + 0.8 * Math.random();

    const x = randomRadius * Math.cos(randomAngle1);
    const y = randomRadius * Math.sin(randomAngle1);

    const speed = 400.0 * randomRadius;
    const vx = -speed * Math.sin(randomAngle1);
    const vy = speed * Math.cos(randomAngle1);

    // pos
    initialParticleData[structSize * i + 0] = x;
    initialParticleData[structSize * i + 1] = y;
    // vel
    initialParticleData[structSize * i + 2] = vx;
    initialParticleData[structSize * i + 3] = vy;
  }

  const particlesBufferA = device.createBuffer({
    label: "Particle buffer A",
    size: initialParticleData.byteLength,
    usage:
      GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(particlesBufferA, 0, initialParticleData);

  const particlesBufferB = device.createBuffer({
    label: "Particle buffer B",
    size: initialParticleData.byteLength,
    usage:
      GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const querySet = device.createQuerySet({ type: "timestamp", count: 4 });
  const queryBuffer = device.createBuffer({
    size: querySet.count * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const stagingBuffer = device.createBuffer({
    size: querySet.count * 8,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
    ],
  });

  const bindGroupA = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: particlesBufferA,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: particlesBufferB,
        },
      },
    ],
  });

  const bindGroupB = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: particlesBufferB,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: particlesBufferA,
        },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  const cShader = `
    struct Particle {
        pos: vec2<f32>,
        vel: vec2<f32>,
    }

    const N = ${numParticles}u;

    @group(0) @binding(0) var<storage, read> input_particles: array<Particle>;
    @group(0) @binding(1) var<storage, read_write> output_particles: array<Particle>;

    var<workgroup> tile: array<Particle, ${WORKGROUP_TILE_SIZE}>;

    const TILE_SIZE = ${WORKGROUP_TILE_SIZE}u;

    @compute @workgroup_size(${WORKGROUP_TILE_SIZE})
    fn main(
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
    }`;

  const cShader2 = `
    struct Particle {
        pos: vec2<f32>,
        vel: vec2<f32>,
    }

    const N = ${numParticles}u;

    @group(0) @binding(0) var<storage, read> input_particles: array<Particle>;
    @group(0) @binding(1) var<storage, read_write> output_particles: array<Particle>;

    @compute @workgroup_size(${WORKGROUP_TILE_SIZE})
    fn main(
      @builtin(global_invocation_id) global_id: vec3<u32>,
    ) {
        let i = global_id.x;
        if (i >= N) {
            return;
        }

        let current_particle = input_particles[i];
        var force = vec2<f32>(0.0, 0.0);

        let dt = 0.00001;
        let softening = 0.01;

        for (var j = 0u; j < N; j = j + 1u) {
            if (j == i) {
                continue;
            }

            let other_particle = input_particles[j];

            let dp = other_particle.pos - current_particle.pos;
            let dist_sq = dot(dp, dp);
            let inv_dist = 1.0 / sqrt(dist_sq + softening);
            let inv_dist_cubed = inv_dist * inv_dist * inv_dist;
            force += dp * inv_dist_cubed;
        }

        var new_vel = current_particle.vel + dt * force;
        var new_pos = current_particle.pos + dt * new_vel;

        output_particles[i].pos = new_pos;
        output_particles[i].vel = new_vel;
    }`;

  const computeShaderModule = device.createShaderModule({
    label: "Compute shader",
    code: cShader2,
  });

  const computePipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: computeShaderModule,
      entryPoint: "main",
    },
  });

  const renderShaderModule = device.createShaderModule({
    label: "Render shader",
    code: `
            // enable f16;

            const particle_size = 0.0005;
            const quad_offsets = array<vec2<f32>, 6>(
                vec2<f32>(-particle_size, -particle_size),
                vec2<f32>( particle_size, -particle_size),
                vec2<f32>(-particle_size,  particle_size),
                vec2<f32>(-particle_size,  particle_size),
                vec2<f32>( particle_size, -particle_size),
                vec2<f32>( particle_size,  particle_size)
            );

            @vertex
            fn vs_main(@location(0) in_pos: vec2<f32>, @builtin(vertex_index) vertex_id : u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(in_pos + quad_offsets[vertex_id], 0.0, 1.0);
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                return vec4<f32>(1.0, 1.0, 1.0, 1.0);
            }
        `,
  });

  const renderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [],
    }),
    vertex: {
      module: renderShaderModule,
      entryPoint: "vs_main",
      buffers: [
        {
          arrayStride: structSize * 4,
          stepMode: "instance",
          attributes: [
            {
              shaderLocation: 0,
              offset: 0,
              format: "float32x2",
            },
          ],
        },
      ],
    },
    fragment: {
      module: renderShaderModule,
      entryPoint: "fs_main",
      targets: [
        {
          format: canvasFormat,
        },
      ],
    },
    primitive: {
      topology: "line-list",
    },
  });

  let frameNum = 0;

  async function frame() {
    const encoder = device.createCommandEncoder();

    const computePass = encoder.beginComputePass({
      timestampWrites: {
        querySet,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      },
    });
    computePass.setPipeline(computePipeline);
    if (frameNum % 2 === 0) {
      computePass.setBindGroup(0, bindGroupA);
    } else {
      computePass.setBindGroup(0, bindGroupB);
    }
    computePass.dispatchWorkgroups(
      Math.ceil(numParticles / WORKGROUP_TILE_SIZE),
    );
    computePass.end();

    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        },
      ],
      timestampWrites: {
        querySet,
        beginningOfPassWriteIndex: 2,
        endOfPassWriteIndex: 3,
      },
    });
    renderPass.setPipeline(renderPipeline);
    if (frameNum % 2 === 0) {
      renderPass.setVertexBuffer(0, particlesBufferB);
    } else {
      renderPass.setVertexBuffer(0, particlesBufferA);
    }
    renderPass.draw(6, numParticles);
    renderPass.end();

    encoder.resolveQuerySet(querySet, 0, querySet.count, queryBuffer, 0);
    encoder.copyBufferToBuffer(
      queryBuffer,
      0,
      stagingBuffer,
      0,
      stagingBuffer.size,
    );

    device.queue.submit([encoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const timings = new BigInt64Array(stagingBuffer.getMappedRange());
    const computeTime = Number(timings[1] - timings[0]) / 1_000_000;
    const renderTime = Number(timings[3] - timings[2]) / 1_000_000;
    stagingBuffer.unmap();

    timingsData.compute.push(computeTime);
    timingsData.render.push(renderTime);
    if (timingsData.compute.length > 10) {
      timingsData.compute.shift();
    }
    if (timingsData.render.length > 10) {
      timingsData.render.shift();
    }
    const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;
    const computeTimeAvg = avg(timingsData.compute);
    const renderTimeAvg = avg(timingsData.render);
    timingInfoEl.textContent = `Compute: ${computeTimeAvg.toFixed(3)}ms | Render: ${renderTimeAvg.toFixed(3)}ms`;

    frameNum++;

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

main().catch((err) => {
  console.error(err);
});

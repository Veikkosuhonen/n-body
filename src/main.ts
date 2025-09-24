import "./style.css";
import rawShaders from "./shaders.wgsl?raw";
import { defineValues } from "./util";

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
    requiredFeatures: ["timestamp-query"],
    requiredLimits: {
      // maxComputeWorkgroupSizeX: 1024,
      // maxComputeInvocationsPerWorkgroup: 1024,
    },
  });

  const context = canvasEl.getContext("webgpu")!;
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
    alphaMode: "premultiplied",
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

  const numParticles = 50_000;
  const WORKGROUP_SIZE = 256;
  const shaders = defineValues(rawShaders, {
    N: numParticles,
    WORKGROUP_SIZE,
    TILE_SIZE: WORKGROUP_SIZE,
  });

  const structSize = 2 * 2; // floats

  const initialParticleData = new Float32Array(numParticles * structSize);
  for (let i = 0; i < numParticles; ++i) {
    const randomAngle1 = 2 * Math.PI * Math.random();
    const randomRadius = 0.1 + 0.7 * Math.random();

    const x = randomRadius * Math.cos(randomAngle1);
    const y = randomRadius * Math.sin(randomAngle1);

    const speed = 220.0 * randomRadius + 10;
    const vx = -speed * Math.sin(randomAngle1);
    const vy = speed * Math.cos(randomAngle1);

    // pos
    initialParticleData[structSize * i + 0] = x;
    initialParticleData[structSize * i + 1] = y;
    // vel
    initialParticleData[structSize * i + 2] = vx;
    initialParticleData[structSize * i + 3] = vy;
    // mass
    // initialParticleData[structSize * i + 4] = 1.0;
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

  const computeShaderModule = device.createShaderModule({
    label: "Compute shader",
    code: shaders,
  });

  const computePipeline = device.createComputePipeline({
    layout: pipelineLayout,
    compute: {
      module: computeShaderModule,
      entryPoint: "n_body_sim_main",
    },
  });

  const renderShaderModule = device.createShaderModule({
    label: "Render shader",
    code: shaders,
  });

  const renderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [],
    }),
    vertex: {
      module: renderShaderModule,
      entryPoint: "particle_vs_main",
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
            {
              shaderLocation: 1,
              offset: 2 * 4, // pos
              format: "float32x2",
            },
            /*{
              shaderLocation: 2,
              offset: 2 * 2 * 4, // pos + vel
              format: "float32",
            },*/
          ],
        },
      ],
    },
    fragment: {
      module: renderShaderModule,
      entryPoint: "particle_fs_main",
      targets: [
        {
          format: canvasFormat,
          blend: {
            color: {
              operation: "add",
              srcFactor: "one",
              dstFactor: "one",
            },
            alpha: {
              operation: "add",
              srcFactor: "one",
              dstFactor: "one",
            },
          },
        },
      ],
    },
    primitive: {
      topology: "triangle-list",
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
    computePass.dispatchWorkgroups(Math.ceil(numParticles / WORKGROUP_SIZE));
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

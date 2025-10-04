import {
  makeShaderDataDefinitions,
  makeStructuredView,
} from 'webgpu-utils';

import "./style.css";
import renderShaders from "./render.wgsl?raw";
import rawComputeShaders from "./compute.wgsl?raw";
import { defineValues } from "./util";
import { Camera } from "./camera";
import { vec2, vec3 } from 'webgpu-matrix';

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
  });

  const context = canvasEl.getContext("webgpu")!;
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: canvasFormat,
    alphaMode: "premultiplied",
  });

  const camera = new Camera();
  camera.zoom = 50.0;
  camera.pan = vec3.fromValues(window.innerWidth/2, window.innerHeight/2, 0);

  let isDragging = false;
  canvasEl.addEventListener("mousedown", (e) => {
    if (e.button === 0) {
      isDragging = true;
    }
  });
  canvasEl.addEventListener("mouseup", (e) => {
    if (e.button === 0) {
      isDragging = false;
    }
  });
  canvasEl.addEventListener("mousemove", (e) => {
    if (isDragging) {
      console.log(e.movementX, e.movementY, camera.zoom)
      const pan = vec3.fromValues(
        (2 * e.movementX),
        (2 * e.movementY),
        0
      );
      // vec3.scale(pan, 1 / camera.zoom, pan);
      vec3.add(camera.pan, pan, camera.pan);
    }
  });
  canvasEl.addEventListener("wheel", (e) => {
    e.preventDefault();
    e.stopPropagation();
    // Zoom towards cursor
    const zoomFactor = e.deltaY < 0 ? 1.01 : 1 / 1.01;
    camera.zoom *= zoomFactor;
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

  const numParticles = 40_000;
  const WORKGROUP_SIZE = 256;
  const computeShaders = defineValues(rawComputeShaders, {
    N: numParticles,
    WORKGROUP_SIZE,
    TILE_SIZE: WORKGROUP_SIZE,
  });

  const computeShaderDefs = makeShaderDataDefinitions(computeShaders);
  const renderShaderDefs = makeShaderDataDefinitions(renderShaders);
  const globalsUniformView = makeStructuredView(renderShaderDefs.uniforms.globals);
  console.log(renderShaderDefs)

  const structSize = 2 * 2; // floats

  const initialParticleData = new Float32Array(numParticles * structSize);
  for (let i = 0; i < numParticles; ++i) {
    const randomAngle1 = 2 * Math.PI * Math.random();
    const randomRadius = 0.3 + 0.6 * Math.random();

    const x = 10 * randomRadius * Math.cos(randomAngle1);
    const y = 10 * randomRadius * Math.sin(randomAngle1);

    const speed = 15 + 40.0 * randomRadius;
    const vx = -speed * Math.sin(randomAngle1);
    const vy = speed * Math.cos(randomAngle1);

    initialParticleData[structSize * i + 0] = x;
    initialParticleData[structSize * i + 1] = y;
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

  const globalsBuffer = device.createBuffer({
    label: "Globals buffer",
    size: globalsUniformView.arrayBuffer.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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

  const computeBindGroupLayout = device.createBindGroupLayout({
    label: "Compute bind group",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });

  const renderBindGroupLayout = device.createBindGroupLayout({
    label: "Render bind group",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" },
      },
    ],
  });

  const bindGroupA = device.createBindGroup({
    label: "Compute bind group A",
    layout: computeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: particlesBufferA } },
      { binding: 1, resource: { buffer: particlesBufferB } },
    ],
  });

  const bindGroupB = device.createBindGroup({
    label: "Bind group B",
    layout: computeBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: particlesBufferB } },
      { binding: 1, resource: { buffer: particlesBufferA } },
    ],
  });

  const renderBindGroup = device.createBindGroup({
    label: "Render bind group",
    layout: renderBindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: globalsBuffer } }],
  });

  const computePipelineLayout = device.createPipelineLayout({
    label: "Compute pipeline layout",
    bindGroupLayouts: [computeBindGroupLayout],
  });

  const renderPipelineLayout = device.createPipelineLayout({
    label: "Render pipeline layout",
    bindGroupLayouts: [renderBindGroupLayout],
  });

  const computeShaderModule = device.createShaderModule({
    label: "Compute shader",
    code: computeShaders,
  });

  const computePipeline = device.createComputePipeline({
    label: "Compute pipeline",
    layout: computePipelineLayout,
    compute: {
      module: computeShaderModule,
      entryPoint: "n_body_sim_tiled_main",
    },
  });

  const renderShaderModule = device.createShaderModule({
    label: "Render shader",
    code: renderShaders,
  });

  const renderPipeline = device.createRenderPipeline({
    label: "Render pipeline",
    layout: renderPipelineLayout,
    vertex: {
      module: renderShaderModule,
      entryPoint: "particle_vs_main",
      buffers: [
        {
          arrayStride: structSize * 4,
          stepMode: "instance",
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x2" },
            { shaderLocation: 1, offset: 2 * 4, format: "float32x2" },
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
    const viewportSize = vec2.fromValues(canvasEl.width, canvasEl.height);
    const viewMatrix = camera.getViewMatrix(viewportSize);
    globalsUniformView.set({
      view_matrix: viewMatrix,
    });
    // console.log(globalsUniformView.arrayBuffer)
    device.queue.writeBuffer(globalsBuffer, 0, globalsUniformView.arrayBuffer);

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
    renderPass.setBindGroup(0, renderBindGroup);
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

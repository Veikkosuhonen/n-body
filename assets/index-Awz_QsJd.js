(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))p(r);new MutationObserver(r=>{for(const t of r)if(t.type==="childList")for(const s of t.addedNodes)s.tagName==="LINK"&&s.rel==="modulepreload"&&p(s)}).observe(document,{childList:!0,subtree:!0});function _(r){const t={};return r.integrity&&(t.integrity=r.integrity),r.referrerPolicy&&(t.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?t.credentials="include":r.crossOrigin==="anonymous"?t.credentials="omit":t.credentials="same-origin",t}function p(r){if(r.ep)return;r.ep=!0;const t=_(r);fetch(r.href,t)}})();const z=document.querySelector("canvas"),D=document.querySelector("#timing-info"),c={compute:[],render:[]};async function N(){if(!navigator.gpu)throw new Error("WebGPU not supported on this browser.");const d=await navigator.gpu.requestAdapter();if(!d)throw new Error("No appropriate GPUAdapter found.");const e=await d.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:{}}),_=z.getContext("webgpu"),p=navigator.gpu.getPreferredCanvasFormat();_.configure({device:e,format:p}),new ResizeObserver(i=>{for(const n of i){const o=n.target,a=n.contentBoxSize[0].inlineSize,g=n.contentBoxSize[0].blockSize;o.width=Math.max(1,Math.min(a,e.limits.maxTextureDimension2D)),o.height=Math.max(1,Math.min(g,e.limits.maxTextureDimension2D))}}).observe(z);const t=5e4,s=256,l=4,u=new Float32Array(t*l);for(let i=0;i<t;++i){const n=2*Math.PI*Math.random(),o=.1+.8*Math.random(),a=o*Math.cos(n),g=o*Math.sin(n),b=400*o,P=-b*Math.sin(n),B=b*Math.cos(n);u[l*i+0]=a,u[l*i+1]=g,u[l*i+2]=P,u[l*i+3]=B}const v=e.createBuffer({label:"Particle buffer A",size:u.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(v,0,u);const h=e.createBuffer({label:"Particle buffer B",size:u.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),f=e.createQuerySet({type:"timestamp",count:4}),S=e.createBuffer({size:f.count*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),m=e.createBuffer({size:f.count*8,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),y=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),E=e.createBindGroup({layout:y,entries:[{binding:0,resource:{buffer:v}},{binding:1,resource:{buffer:h}}]}),M=e.createBindGroup({layout:y,entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:v}}]}),O=e.createPipelineLayout({bindGroupLayouts:[y]}),L=`
    struct Particle {
        pos: vec2<f32>,
        vel: vec2<f32>,
    }

    const N = ${t}u;

    @group(0) @binding(0) var<storage, read> input_particles: array<Particle>;
    @group(0) @binding(1) var<storage, read_write> output_particles: array<Particle>;

    var<workgroup> tile: array<Particle, ${s}>;

    const TILE_SIZE = ${s}u;

    @compute @workgroup_size(${s})
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
    }`,T=e.createShaderModule({label:"Compute shader",code:L}),A=e.createComputePipeline({layout:O,compute:{module:T,entryPoint:"main"}}),w=e.createShaderModule({label:"Render shader",code:`
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
        `}),q=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[]}),vertex:{module:w,entryPoint:"vs_main",buffers:[{arrayStride:l*4,stepMode:"instance",attributes:[{shaderLocation:0,offset:0,format:"float32x2"}]}]},fragment:{module:w,entryPoint:"fs_main",targets:[{format:p}]},primitive:{topology:"line-list"}});let x=0;async function U(){const i=e.createCommandEncoder(),n=i.beginComputePass({timestampWrites:{querySet:f,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}});n.setPipeline(A),x%2===0?n.setBindGroup(0,E):n.setBindGroup(0,M),n.dispatchWorkgroups(Math.ceil(t/s)),n.end();const o=i.beginRenderPass({colorAttachments:[{view:_.getCurrentTexture().createView(),loadOp:"clear",clearValue:{r:0,g:0,b:0,a:1},storeOp:"store"}],timestampWrites:{querySet:f,beginningOfPassWriteIndex:2,endOfPassWriteIndex:3}});o.setPipeline(q),x%2===0?o.setVertexBuffer(0,h):o.setVertexBuffer(0,v),o.draw(6,t),o.end(),i.resolveQuerySet(f,0,f.count,S,0),i.copyBufferToBuffer(S,0,m,0,m.size),e.queue.submit([i.finish()]),await m.mapAsync(GPUMapMode.READ);const a=new BigInt64Array(m.getMappedRange()),g=Number(a[1]-a[0])/1e6,b=Number(a[3]-a[2])/1e6;m.unmap(),c.compute.push(g),c.render.push(b),c.compute.length>10&&c.compute.shift(),c.render.length>10&&c.render.shift();const P=G=>G.reduce((I,C)=>I+C,0)/G.length,B=P(c.compute),R=P(c.render);D.textContent=`Compute: ${B.toFixed(3)}ms | Render: ${R.toFixed(3)}ms`,x++,requestAnimationFrame(U)}requestAnimationFrame(U)}N().catch(d=>{console.error(d)});

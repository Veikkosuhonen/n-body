(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const r of document.querySelectorAll('link[rel="modulepreload"]'))p(r);new MutationObserver(r=>{for(const t of r)if(t.type==="childList")for(const u of t.addedNodes)u.tagName==="LINK"&&u.rel==="modulepreload"&&p(u)}).observe(document,{childList:!0,subtree:!0});function v(r){const t={};return r.integrity&&(t.integrity=r.integrity),r.referrerPolicy&&(t.referrerPolicy=r.referrerPolicy),r.crossOrigin==="use-credentials"?t.credentials="include":r.crossOrigin==="anonymous"?t.credentials="omit":t.credentials="same-origin",t}function p(r){if(r.ep)return;r.ep=!0;const t=v(r);fetch(r.href,t)}})();const z=document.querySelector("canvas"),I=document.querySelector("#timing-info"),a={compute:[],render:[]};async function N(){if(!navigator.gpu)throw new Error("WebGPU not supported on this browser.");const d=await navigator.gpu.requestAdapter();if(!d)throw new Error("No appropriate GPUAdapter found.");const e=await d.requestDevice({requiredFeatures:["timestamp-query"],requiredLimits:{}}),v=z.getContext("webgpu"),p=navigator.gpu.getPreferredCanvasFormat();v.configure({device:e,format:p}),new ResizeObserver(i=>{for(const n of i){const o=n.target,s=n.contentBoxSize[0].inlineSize,g=n.contentBoxSize[0].blockSize;o.width=Math.max(1,Math.min(s,e.limits.maxTextureDimension2D)),o.height=Math.max(1,Math.min(g,e.limits.maxTextureDimension2D))}}).observe(z);const t=5e4,u=256,f=4,c=new Float32Array(t*f);for(let i=0;i<t;++i){const n=2*Math.PI*Math.random(),o=.1+.8*Math.random(),s=o*Math.cos(n),g=o*Math.sin(n),b=400*o,P=-b*Math.sin(n),x=b*Math.cos(n);c[f*i+0]=s,c[f*i+1]=g,c[f*i+2]=P,c[f*i+3]=x}const _=e.createBuffer({label:"Particle buffer A",size:c.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});e.queue.writeBuffer(_,0,c);const h=e.createBuffer({label:"Particle buffer B",size:c.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),l=e.createQuerySet({type:"timestamp",count:4}),S=e.createBuffer({size:l.count*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),m=e.createBuffer({size:l.count*8,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST}),y=e.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),M=e.createBindGroup({layout:y,entries:[{binding:0,resource:{buffer:_}},{binding:1,resource:{buffer:h}}]}),O=e.createBindGroup({layout:y,entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:_}}]}),E=e.createPipelineLayout({bindGroupLayouts:[y]}),A=`
    struct Particle {
        pos: vec2<f32>,
        vel: vec2<f32>,
    }

    const N = ${t}u;

    @group(0) @binding(0) var<storage, read> input_particles: array<Particle>;
    @group(0) @binding(1) var<storage, read_write> output_particles: array<Particle>;

    @compute @workgroup_size(${u})
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
    }`,q=e.createShaderModule({label:"Compute shader",code:A}),L=e.createComputePipeline({layout:E,compute:{module:q,entryPoint:"main"}}),U=e.createShaderModule({label:"Render shader",code:`
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
        `}),R=e.createRenderPipeline({layout:e.createPipelineLayout({bindGroupLayouts:[]}),vertex:{module:U,entryPoint:"vs_main",buffers:[{arrayStride:f*4,stepMode:"instance",attributes:[{shaderLocation:0,offset:0,format:"float32x2"}]}]},fragment:{module:U,entryPoint:"fs_main",targets:[{format:p}]},primitive:{topology:"line-list"}});let B=0;async function G(){const i=e.createCommandEncoder(),n=i.beginComputePass({timestampWrites:{querySet:l,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}});n.setPipeline(L),B%2===0?n.setBindGroup(0,M):n.setBindGroup(0,O),n.dispatchWorkgroups(Math.ceil(t/u)),n.end();const o=i.beginRenderPass({colorAttachments:[{view:v.getCurrentTexture().createView(),loadOp:"clear",clearValue:{r:0,g:0,b:0,a:1},storeOp:"store"}],timestampWrites:{querySet:l,beginningOfPassWriteIndex:2,endOfPassWriteIndex:3}});o.setPipeline(R),B%2===0?o.setVertexBuffer(0,h):o.setVertexBuffer(0,_),o.draw(6,t),o.end(),i.resolveQuerySet(l,0,l.count,S,0),i.copyBufferToBuffer(S,0,m,0,m.size),e.queue.submit([i.finish()]),await m.mapAsync(GPUMapMode.READ);const s=new BigInt64Array(m.getMappedRange()),g=Number(s[1]-s[0])/1e6,b=Number(s[3]-s[2])/1e6;m.unmap(),a.compute.push(g),a.render.push(b),a.compute.length>10&&a.compute.shift(),a.render.length>10&&a.render.shift();const P=w=>w.reduce((C,D)=>C+D,0)/w.length,x=P(a.compute),T=P(a.render);I.textContent=`Compute: ${x.toFixed(3)}ms | Render: ${T.toFixed(3)}ms`,B++,requestAnimationFrame(G)}requestAnimationFrame(G)}N().catch(d=>{console.error(d)});

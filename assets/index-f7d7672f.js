(function(){const i=document.createElement("link").relList;if(i&&i.supports&&i.supports("modulepreload"))return;for(const t of document.querySelectorAll('link[rel="modulepreload"]'))n(t);new MutationObserver(t=>{for(const s of t)if(s.type==="childList")for(const a of s.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&n(a)}).observe(document,{childList:!0,subtree:!0});function r(t){const s={};return t.integrity&&(s.integrity=t.integrity),t.referrerPolicy&&(s.referrerPolicy=t.referrerPolicy),t.crossOrigin==="use-credentials"?s.credentials="include":t.crossOrigin==="anonymous"?s.credentials="omit":s.credentials="same-origin",s}function n(t){if(t.ep)return;t.ep=!0;const s=r(t);fetch(t.href,s)}})();const X=`
struct Matrix {
    numbers: array<f32>
};

struct Sizes {
    sizeX: u32,
    sizeY: u32,
    dt: f32
}

@group(0) @binding(0) var<storage, read> inXMatrix:  Matrix;
@group(0) @binding(1) var<storage, read_write> outXMatrix: Matrix;
@group(0) @binding(2) var<storage, read> inVMatrix: Matrix;
@group(0) @binding(3) var<storage, read_write> outVMatrix: Matrix;
@group(0) @binding(4) var<storage, read> inCMatrix: Matrix;
@group(0) @binding(5) var<uniform> inSizes: Sizes;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Add bounds check
    if (global_id.x >= inSizes.sizeX || global_id.y >= inSizes.sizeY) {
        return;
    }

    let xIndex = global_id.x;
    let yIndex = global_id.y;
    let index = xIndex + yIndex * inSizes.sizeX;


    var xValue = inXMatrix.numbers[index];
    var vValue = inVMatrix.numbers[index];
    var cValue = inCMatrix.numbers[index];

    let sizeX = i32(inSizes.sizeX);
    let sizeY = i32(inSizes.sizeY);
    let idx = i32(index);
    let x = i32(xIndex);
    let y = i32(yIndex);


    if (x > 0) {
        let n = idx - 1;
        vValue += (inXMatrix.numbers[u32(n)] - xValue) * cValue;
    }

    if (x < sizeX - 1) {
        let n = idx + 1;
        vValue += (inXMatrix.numbers[u32(n)] - xValue) * cValue;
    }

    if (y > 0) {
        let n = idx - sizeX;
        vValue += (inXMatrix.numbers[u32(n)] - xValue) * cValue;
    }

    if (y < sizeY - 1) {
        let n = idx + sizeX;
        vValue += (inXMatrix.numbers[u32(n)] - xValue) * cValue;
    }

    outVMatrix.numbers[index] = vValue;
    outXMatrix.numbers[index] = xValue + inSizes.dt * vValue; 

    if (cValue == 0.0) {
        outXMatrix.numbers[index] = 0.0;
        outVMatrix.numbers[index] = 0.0;

    }

}
`;class A{constructor(){this.current=0,this.XSize=0,this.YSize=0}async loadOnGpu(i,r,n,t,s,a){if(!("gpu"in navigator))throw new Error("Your browser doesn't support WebGPU. Please use Chrome version 113+ or Edge version 113+.");const o=await navigator.gpu.requestAdapter();if(!o)throw new Error("Failed to obtain GPU adapter.");this.device=await o.requestDevice(),this.XSize=i,this.YSize=r;const d=i*r*4,l=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,m=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;this.buffers={X:[this.device.createBuffer({size:d,usage:l,mappedAtCreation:!0}),this.device.createBuffer({size:d,usage:m})],V:[this.device.createBuffer({size:d,usage:l,mappedAtCreation:!0}),this.device.createBuffer({size:d,usage:m})],C:this.device.createBuffer({size:d,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,mappedAtCreation:!0})},new Float32Array(this.buffers.X[0].getMappedRange()).set(t),new Float32Array(this.buffers.V[0].getMappedRange()).set(s),new Float32Array(this.buffers.C.getMappedRange()).set(a),this.buffers.X[0].unmap(),this.buffers.V[0].unmap(),this.buffers.C.unmap();const g=new ArrayBuffer(16);new Uint32Array(g,0,2).set([i,r]),new Float32Array(g,8,1).set([n]),this.uniformBuffer=this.device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(this.uniformBuffer,0,g);const S=this.device.createShaderModule({code:X});this.pipeline=this.device.createComputePipeline({layout:"auto",compute:{module:S,entryPoint:"main"}}),this.bindGroups=[0,1].map(b=>this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.buffers.X[b]}},{binding:1,resource:{buffer:this.buffers.X[1-b]}},{binding:2,resource:{buffer:this.buffers.V[b]}},{binding:3,resource:{buffer:this.buffers.V[1-b]}},{binding:4,resource:{buffer:this.buffers.C}},{binding:5,resource:{buffer:this.uniformBuffer}}]}))}update(){const i=this.device.createCommandEncoder(),r=i.beginComputePass();r.setPipeline(this.pipeline),r.setBindGroup(0,this.bindGroups[this.current]),r.dispatchWorkgroups(Math.ceil(this.XSize/16),Math.ceil(this.YSize/16)),r.end(),this.device.queue.submit([i.finish()]),this.current=1-this.current}async readOnCpu(){const i=this.XSize*this.YSize*4,r=this.device.createBuffer({size:i,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),n=this.device.createBuffer({size:i,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),t=this.device.createCommandEncoder();t.copyBufferToBuffer(this.buffers.X[this.current],0,r,0,i),t.copyBufferToBuffer(this.buffers.V[this.current],0,n,0,i),this.device.queue.submit([t.finish()]),await Promise.all([r.mapAsync(GPUMapMode.READ),n.mapAsync(GPUMapMode.READ)]);const s=new Float32Array(r.getMappedRange().slice()),a=new Float32Array(n.getMappedRange().slice());return r.unmap(),n.unmap(),{X:s,V:a}}async readXOnCpu(){const i=this.XSize*this.YSize*4,r=this.device.createBuffer({size:i,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),n=this.device.createCommandEncoder();n.copyBufferToBuffer(this.buffers.X[this.current],0,r,0,i),this.device.queue.submit([n.finish()]),await r.mapAsync(GPUMapMode.READ);const t=new Float32Array(r.getMappedRange().slice());return r.unmap(),t}changeCSpace(i){if(i.length!==this.XSize*this.YSize)throw new Error(`Expected newCSpace length ${this.XSize*this.YSize}, got ${i.length}`);this.device.queue.writeBuffer(this.buffers.C,0,i)}}function M(e,i,r){const n=[];for(let t=0;t<e*i;t++)n.push(r);return n}function x(e){return 1/(1+Math.exp(-e))}var f=[];function P(e,i,r,n){const t=document.getElementById("waveCanvas");t.tabIndex=0,t.style.outline="none",t.focus();const s=t.getContext("2d"),a=s.createImageData(i,r);for(let o=0;o<e.length;o++){if(n[o]===0){a.data[o*4]=0,a.data[o*4+1]=0,a.data[o*4+2]=0,a.data[o*4+3]=0;continue}const d=Math.floor(x(e[o]+n[o])*255),l=Math.floor(x(e[o]-n[o])*255),m=Math.floor(x(e[o])*255),g=o*4;a.data[g]=d,a.data[g+1]=l,a.data[g+2]=m,a.data[g+3]=255}s.putImageData(a,0,0)}function w(e,i,r,n,t,s,a,o){for(let d=0;d<s;d++)for(let l=0;l<a;l++)d>=e&&d<=r&&l>=i&&l<=n&&(o[d+l*s]=t);return o}const u=700,c=700,B=1/60,C=16,G=B*Math.sqrt(C),V=B/Math.sqrt(C);var v=new A;let h=M(u,c,0);h=w(u/2,c/2,u/2,c/2,300,u,c,h);let _=M(u,c,0);var y=0;f=M(u,c,1);for(let e=0;e<u;e++)f[e]=0,f[e+(c-1)*u]=0;for(let e=0;e<c;e++)f[e*u]=0,f[(e+1)*u-1]=0;console.log("GPU initialized");async function E(){const e=document.getElementById("waveCanvas");try{await v.loadOnGpu(u,c,G,new Float32Array(h),new Float32Array(_),new Float32Array(f)),e.width=u,e.height=c,P(h,u,c,f),setInterval(async()=>{await v.update(),h=Array.from(await v.readXOnCpu()),P(h,u,c,f)},1e3*V)}catch(i){const r=e.getContext("2d");r.font="24px Arial",r.fillStyle="red",r.textAlign="center",r.fillText("This demo requires WebGPU support.",e.width/2,e.height/2-50),r.fillText("Please use Chrome 113+ or Edge 113+.",e.width/2,e.height/2),console.error("WebGPU error:",i)}}const p=document.getElementById("waveCanvas"),U=async e=>{const i=p.getBoundingClientRect(),r=e.clientX-i.left,n=e.clientY-i.top,t=Math.floor(r*u/i.width),s=Math.floor(n*c/i.height);f=w(t-10,s-10,t+10,s+10,y,u,c,f),v.changeCSpace(new Float32Array(f)),console.log("CSpace changed")};p.addEventListener("click",U);var z=!1;p.addEventListener("mousedown",async e=>{z=!0});p.addEventListener("mouseup",async e=>{z=!1});p.addEventListener("mousemove",async e=>{z&&await U(e)});p.addEventListener("keydown",async e=>{["S","s","ы","Ы"].includes(e.key)?(y=.5-y,console.log("CFill updated to:",y)):console.log("Key pressed:",e.key)});E().catch(console.error);

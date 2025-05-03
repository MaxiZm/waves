(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const t of document.querySelectorAll('link[rel="modulepreload"]'))a(t);new MutationObserver(t=>{for(const r of t)if(r.type==="childList")for(const s of r.addedNodes)s.tagName==="LINK"&&s.rel==="modulepreload"&&a(s)}).observe(document,{childList:!0,subtree:!0});function i(t){const r={};return t.integrity&&(r.integrity=t.integrity),t.referrerPolicy&&(r.referrerPolicy=t.referrerPolicy),t.crossOrigin==="use-credentials"?r.credentials="include":t.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function a(t){if(t.ep)return;t.ep=!0;const r=i(t);fetch(t.href,r)}})();const U=`
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

@compute @workgroup_size(32, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
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


    if (x >= 0) {
        let n = idx - 1;
        vValue += (inXMatrix.numbers[u32(n)] - xValue) * cValue;
    }

    if (x <= sizeX - 1) {
        let n = idx + 1;
        vValue += (inXMatrix.numbers[u32(n)] - xValue) * cValue;
    }

    if (y >= 0) {
        let n = idx - sizeX;
        vValue += (inXMatrix.numbers[u32(n)] - xValue) * cValue;
    }

    if (y <= sizeY - 1) {
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
`;class S{constructor(){this.current=0,this.XSize=0,this.YSize=0}async loadOnGpu(e,i,a,t,r,s){if(!navigator.gpu)throw new Error("WebGPU not supported in this browser.");const u=await navigator.gpu.requestAdapter();if(!u)throw new Error("Failed to obtain GPU adapter.");this.device=await u.requestDevice(),this.XSize=e,this.YSize=i;const o=e*i*4,f=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,m=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;this.buffers={X:[this.device.createBuffer({size:o,usage:f,mappedAtCreation:!0}),this.device.createBuffer({size:o,usage:m})],V:[this.device.createBuffer({size:o,usage:f,mappedAtCreation:!0}),this.device.createBuffer({size:o,usage:m})],C:this.device.createBuffer({size:o,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,mappedAtCreation:!0})},new Float32Array(this.buffers.X[0].getMappedRange()).set(t),new Float32Array(this.buffers.V[0].getMappedRange()).set(r),new Float32Array(this.buffers.C.getMappedRange()).set(s),this.buffers.X[0].unmap(),this.buffers.V[0].unmap(),this.buffers.C.unmap();const l=new ArrayBuffer(16);new Uint32Array(l,0,2).set([e,i]),new Float32Array(l,8,1).set([a]),this.uniformBuffer=this.device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(this.uniformBuffer,0,l);const w=this.device.createShaderModule({code:U});this.pipeline=this.device.createComputePipeline({layout:"auto",compute:{module:w,entryPoint:"main"}}),this.bindGroups=[0,1].map(v=>this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.buffers.X[v]}},{binding:1,resource:{buffer:this.buffers.X[1-v]}},{binding:2,resource:{buffer:this.buffers.V[v]}},{binding:3,resource:{buffer:this.buffers.V[1-v]}},{binding:4,resource:{buffer:this.buffers.C}},{binding:5,resource:{buffer:this.uniformBuffer}}]}))}update(){const e=this.device.createCommandEncoder(),i=e.beginComputePass();i.setPipeline(this.pipeline),i.setBindGroup(0,this.bindGroups[this.current]),i.dispatchWorkgroups(Math.ceil(this.XSize/32),Math.ceil(this.YSize/8)),i.end(),this.device.queue.submit([e.finish()]),this.current=1-this.current}async readOnCpu(){const e=this.XSize*this.YSize*4,i=this.device.createBuffer({size:e,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),a=this.device.createBuffer({size:e,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),t=this.device.createCommandEncoder();t.copyBufferToBuffer(this.buffers.X[this.current],0,i,0,e),t.copyBufferToBuffer(this.buffers.V[this.current],0,a,0,e),this.device.queue.submit([t.finish()]),await Promise.all([i.mapAsync(GPUMapMode.READ),a.mapAsync(GPUMapMode.READ)]);const r=new Float32Array(i.getMappedRange().slice()),s=new Float32Array(a.getMappedRange().slice());return i.unmap(),a.unmap(),{X:r,V:s}}async readXOnCpu(){const e=this.XSize*this.YSize*4,i=this.device.createBuffer({size:e,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),a=this.device.createCommandEncoder();a.copyBufferToBuffer(this.buffers.X[this.current],0,i,0,e),this.device.queue.submit([a.finish()]),await i.mapAsync(GPUMapMode.READ);const t=new Float32Array(i.getMappedRange().slice());return i.unmap(),t}changeCSpace(e){if(e.length!==this.XSize*this.YSize)throw new Error(`Expected newCSpace length ${this.XSize*this.YSize}, got ${e.length}`);this.device.queue.writeBuffer(this.buffers.C,0,e)}}function y(n,e,i){const a=[];for(let t=0;t<n*e;t++)a.push(i);return a}function x(n){return 1/(1+Math.exp(-n))}var g=[];function B(n,e,i,a){const r=document.getElementById("waveCanvas").getContext("2d"),s=r.createImageData(e,i);for(let u=0;u<n.length;u++){if(a[u]===0){s.data[u*4]=0,s.data[u*4+1]=0,s.data[u*4+2]=0,s.data[u*4+3]=0;continue}const o=Math.floor(x(n[u])*255),f=Math.floor(x(n[u]-1)*255),m=Math.floor(x(n[u]+1)*255),l=u*4;s.data[l]=o,s.data[l+1]=f,s.data[l+2]=m,s.data[l+3]=255}r.putImageData(s,0,0)}function C(n,e,i,a,t,r,s,u){for(let o=0;o<r;o++)for(let f=0;f<s;f++)o>=n&&o<=i&&f>=e&&f<=a&&(u[o+f*r]=t);return u}const c=500,d=500,z=1/60;var b=new S;let p=y(c,d,0);p[c/2+d*c/2]=300;let X=y(c,d,0);g=y(c,d,1);console.log("GPU initialized");async function A(){await b.loadOnGpu(c,d,z,new Float32Array(p),new Float32Array(X),new Float32Array(g)),h.width=c,h.height=d,B(p,c,d,g),setInterval(async()=>{await b.update(),p=Array.from(await b.readXOnCpu()),B(p,c,d,g)},1e3*z)}const h=document.getElementById("waveCanvas"),P=async n=>{const e=h.getBoundingClientRect(),i=n.clientX-e.left,a=n.clientY-e.top,t=Math.floor(i*c/e.width),r=Math.floor(a*d/e.height);g=C(t-10,r-10,t+10,r+10,0,c,d,g),b.changeCSpace(new Float32Array(g)),console.log("CSpace changed")};h.addEventListener("click",P);var M=!1;h.addEventListener("mousedown",async n=>{M=!0});h.addEventListener("mouseup",async n=>{M=!1});h.addEventListener("mousemove",async n=>{M&&await P(n)});A().catch(console.error);

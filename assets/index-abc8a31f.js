(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const e of document.querySelectorAll('link[rel="modulepreload"]'))a(e);new MutationObserver(e=>{for(const s of e)if(s.type==="childList")for(const n of s.addedNodes)n.tagName==="LINK"&&n.rel==="modulepreload"&&a(n)}).observe(document,{childList:!0,subtree:!0});function r(e){const s={};return e.integrity&&(s.integrity=e.integrity),e.referrerPolicy&&(s.referrerPolicy=e.referrerPolicy),e.crossOrigin==="use-credentials"?s.credentials="include":e.crossOrigin==="anonymous"?s.credentials="omit":s.credentials="same-origin",s}function a(e){if(e.ep)return;e.ep=!0;const s=r(e);fetch(e.href,s)}})();const X=`
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
`;class C{constructor(){this.current=0,this.XSize=0,this.YSize=0}async loadOnGpu(t,r,a,e,s,n){if(!navigator.gpu)throw new Error("WebGPU not supported in this browser.");const o=await navigator.gpu.requestAdapter();if(!o)throw new Error("Failed to obtain GPU adapter.");this.device=await o.requestDevice(),this.XSize=t,this.YSize=r;const c=t*r*4,l=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC,m=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;this.buffers={X:[this.device.createBuffer({size:c,usage:l,mappedAtCreation:!0}),this.device.createBuffer({size:c,usage:m})],V:[this.device.createBuffer({size:c,usage:l,mappedAtCreation:!0}),this.device.createBuffer({size:c,usage:m})],C:this.device.createBuffer({size:c,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,mappedAtCreation:!0})},new Float32Array(this.buffers.X[0].getMappedRange()).set(e),new Float32Array(this.buffers.V[0].getMappedRange()).set(s),new Float32Array(this.buffers.C.getMappedRange()).set(n),this.buffers.X[0].unmap(),this.buffers.V[0].unmap(),this.buffers.C.unmap();const g=new ArrayBuffer(16);new Uint32Array(g,0,2).set([t,r]),new Float32Array(g,8,1).set([a]),this.uniformBuffer=this.device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.device.queue.writeBuffer(this.uniformBuffer,0,g);const U=this.device.createShaderModule({code:X});this.pipeline=this.device.createComputePipeline({layout:"auto",compute:{module:U,entryPoint:"main"}}),this.bindGroups=[0,1].map(b=>this.device.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:this.buffers.X[b]}},{binding:1,resource:{buffer:this.buffers.X[1-b]}},{binding:2,resource:{buffer:this.buffers.V[b]}},{binding:3,resource:{buffer:this.buffers.V[1-b]}},{binding:4,resource:{buffer:this.buffers.C}},{binding:5,resource:{buffer:this.uniformBuffer}}]}))}update(){const t=this.device.createCommandEncoder(),r=t.beginComputePass();r.setPipeline(this.pipeline),r.setBindGroup(0,this.bindGroups[this.current]),r.dispatchWorkgroups(Math.ceil(this.XSize/16),Math.ceil(this.YSize/16)),r.end(),this.device.queue.submit([t.finish()]),this.current=1-this.current}async readOnCpu(){const t=this.XSize*this.YSize*4,r=this.device.createBuffer({size:t,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),a=this.device.createBuffer({size:t,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),e=this.device.createCommandEncoder();e.copyBufferToBuffer(this.buffers.X[this.current],0,r,0,t),e.copyBufferToBuffer(this.buffers.V[this.current],0,a,0,t),this.device.queue.submit([e.finish()]),await Promise.all([r.mapAsync(GPUMapMode.READ),a.mapAsync(GPUMapMode.READ)]);const s=new Float32Array(r.getMappedRange().slice()),n=new Float32Array(a.getMappedRange().slice());return r.unmap(),a.unmap(),{X:s,V:n}}async readXOnCpu(){const t=this.XSize*this.YSize*4,r=this.device.createBuffer({size:t,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),a=this.device.createCommandEncoder();a.copyBufferToBuffer(this.buffers.X[this.current],0,r,0,t),this.device.queue.submit([a.finish()]),await r.mapAsync(GPUMapMode.READ);const e=new Float32Array(r.getMappedRange().slice());return r.unmap(),e}changeCSpace(t){if(t.length!==this.XSize*this.YSize)throw new Error(`Expected newCSpace length ${this.XSize*this.YSize}, got ${t.length}`);this.device.queue.writeBuffer(this.buffers.C,0,t)}}function M(i,t,r){const a=[];for(let e=0;e<i*t;e++)a.push(r);return a}function x(i){return 1/(1+Math.exp(-i))}var d=[];function B(i,t,r,a){const e=document.getElementById("waveCanvas");e.tabIndex=0,e.style.outline="none",e.focus();const s=e.getContext("2d"),n=s.createImageData(t,r);for(let o=0;o<i.length;o++){if(a[o]===0){n.data[o*4]=0,n.data[o*4+1]=0,n.data[o*4+2]=0,n.data[o*4+3]=0;continue}const c=Math.floor(x(i[o]+a[o])*255),l=Math.floor(x(i[o]-a[o])*255),m=Math.floor(x(i[o])*255),g=o*4;n.data[g]=c,n.data[g+1]=l,n.data[g+2]=m,n.data[g+3]=255}s.putImageData(n,0,0)}function A(i,t,r,a,e,s,n,o){for(let c=0;c<s;c++)for(let l=0;l<n;l++)c>=i&&c<=r&&l>=t&&l<=a&&(o[c+l*s]=e);return o}const u=500,f=500,P=1/60,w=4,G=P*Math.sqrt(w),V=P/Math.sqrt(w);var v=new C;let p=M(u,f,0);p[u/2+f*u/2]=300;let _=M(u,f,0);var y=0;d=M(u,f,1);for(let i=0;i<u;i++)d[i]=0,d[i+(f-1)*u]=0;for(let i=0;i<f;i++)d[i*u]=0,d[(i+1)*u-1]=0;console.log("GPU initialized");async function O(){await v.loadOnGpu(u,f,G,new Float32Array(p),new Float32Array(_),new Float32Array(d)),h.width=u,h.height=f,B(p,u,f,d),setInterval(async()=>{await v.update(),p=Array.from(await v.readXOnCpu()),B(p,u,f,d)},1e3*V)}const h=document.getElementById("waveCanvas"),S=async i=>{const t=h.getBoundingClientRect(),r=i.clientX-t.left,a=i.clientY-t.top,e=Math.floor(r*u/t.width),s=Math.floor(a*f/t.height);d=A(e-10,s-10,e+10,s+10,y,u,f,d),v.changeCSpace(new Float32Array(d)),console.log("CSpace changed")};h.addEventListener("click",S);var z=!1;h.addEventListener("mousedown",async i=>{z=!0});h.addEventListener("mouseup",async i=>{z=!1});h.addEventListener("mousemove",async i=>{z&&await S(i)});h.addEventListener("keydown",async i=>{["S","s","ы","Ы"].includes(i.key)?(y=.5-y,console.log("CFill updated to:",y)):console.log("Key pressed:",i.key)});O().catch(console.error);

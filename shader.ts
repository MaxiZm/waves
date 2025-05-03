import shaderCode from "./shader_code";

export class WaveSimulator {
  private device!: GPUDevice;
  private pipeline!: GPUComputePipeline;
  private uniformBuffer!: GPUBuffer;
  private bindGroups!: [GPUBindGroup, GPUBindGroup];
  private buffers!: {
    X: [GPUBuffer, GPUBuffer];
    V: [GPUBuffer, GPUBuffer];
    C: GPUBuffer;
  };
  private current = 0;
  private XSize = 0;
  private YSize = 0;

  public async loadOnGpu(
    XSize: number,
    YSize: number,
    dt: number,
    initialX: Float32Array,
    initialV: Float32Array,
    CSpace: Float32Array
  ): Promise<void> {
    if (!navigator.gpu) {
      throw new Error("WebGPU not supported in this browser.");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("Failed to obtain GPU adapter.");
    this.device = await adapter.requestDevice();

    this.XSize = XSize;
    this.YSize = YSize;

    const byteSize = XSize * YSize * 4;
    const usageIn = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const usageOut = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

    this.buffers = {
      X: [
        this.device.createBuffer({ size: byteSize, usage: usageIn, mappedAtCreation: true }),
        this.device.createBuffer({ size: byteSize, usage: usageOut })
      ],
      V: [
        this.device.createBuffer({ size: byteSize, usage: usageIn, mappedAtCreation: true }),
        this.device.createBuffer({ size: byteSize, usage: usageOut })
      ],
      C: this.device.createBuffer({ size: byteSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true })
    };

    new Float32Array(this.buffers.X[0].getMappedRange()).set(initialX);
    new Float32Array(this.buffers.V[0].getMappedRange()).set(initialV);
    new Float32Array(this.buffers.C.getMappedRange()).set(CSpace);
    this.buffers.X[0].unmap();
    this.buffers.V[0].unmap();
    this.buffers.C.unmap();

    const uniformData = new ArrayBuffer(16);
    new Uint32Array(uniformData, 0, 2).set([XSize, YSize]);
    new Float32Array(uniformData, 8, 1).set([dt]);
    this.uniformBuffer = this.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    const module = this.device.createShaderModule({ code: shaderCode });
    this.pipeline = this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });

    this.bindGroups = [0, 1].map(i =>
      this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.buffers.X[i] } },
          { binding: 1, resource: { buffer: this.buffers.X[1 - i] } },
          { binding: 2, resource: { buffer: this.buffers.V[i] } },
          { binding: 3, resource: { buffer: this.buffers.V[1 - i] } },
          { binding: 4, resource: { buffer: this.buffers.C } },
          { binding: 5, resource: { buffer: this.uniformBuffer } }
        ]
      })
    ) as [GPUBindGroup, GPUBindGroup];
  }

  public update(): void {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroups[this.current]);
    pass.dispatchWorkgroups(
      Math.ceil(this.XSize / 32),
      Math.ceil(this.YSize / 8)
    );
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    this.current = 1 - this.current;
  }

  public async readOnCpu(): Promise<{ X: Float32Array; V: Float32Array }> {
    const byteSize = this.XSize * this.YSize * 4;
    const readX = this.device.createBuffer({ size: byteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const readV = this.device.createBuffer({ size: byteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffers.X[this.current], 0, readX, 0, byteSize);
    encoder.copyBufferToBuffer(this.buffers.V[this.current], 0, readV, 0, byteSize);
    this.device.queue.submit([encoder.finish()]);

    await Promise.all([readX.mapAsync(GPUMapMode.READ), readV.mapAsync(GPUMapMode.READ)]);

    const X = new Float32Array(readX.getMappedRange().slice());
    const V = new Float32Array(readV.getMappedRange().slice());
    readX.unmap();
    readV.unmap();
    return { X, V };
  }

  public async readXOnCpu(): Promise<Float32Array> {
    const byteSize = this.XSize * this.YSize * 4;
    const readX = this.device.createBuffer({ size: byteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(this.buffers.X[this.current], 0, readX, 0, byteSize);
    this.device.queue.submit([encoder.finish()]);

    await readX.mapAsync(GPUMapMode.READ);
    const X = new Float32Array(readX.getMappedRange().slice());
    readX.unmap();
    return X;
  }

  public changeCSpace(newCSpace: Float32Array): void {
    if (newCSpace.length !== this.XSize * this.YSize) {
      throw new Error(`Expected newCSpace length ${this.XSize * this.YSize}, got ${newCSpace.length}`);
    }
    this.device.queue.writeBuffer(this.buffers.C, 0, newCSpace);
  }
}

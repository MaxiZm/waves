import { WaveSimulator } from "./shader.ts";

function getSpace(x: number, y: number, c: number): number[] {
    const space: number[] = [];

    for (let i = 0; i < x * y; i++) {
        space.push(c);
    }

    return space;
}

function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

var CSpace: number[] = [];

function renderToCanvas(xSpace: number[], width: number, height: number, CSpace: number[]): void {
    const canvas = document.getElementById('waveCanvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.createImageData(width, height);

    for (let i = 0; i < xSpace.length; i++) {
        if (CSpace[i] === 0) {
            imageData.data[i * 4] = 0;     // R
            imageData.data[i * 4 + 1] = 0; // G
            imageData.data[i * 4 + 2] = 0; // B
            imageData.data[i * 4 + 3] = 0; // A
            continue;
        }

        // Convert wave height to color intensity (assuming values between -1 and 1)
        const valueR = Math.floor(sigmoid(xSpace[i]) * 255);
        const valueG = Math.floor(sigmoid(xSpace[i] - 1) * 255);
        const valueB = Math.floor(sigmoid(xSpace[i] + 1) * 255);
        const index = i * 4;
        imageData.data[index] = valueR;     // R
        imageData.data[index + 1] = valueG; // G
        imageData.data[index + 2] = valueB; // B
        imageData.data[index + 3] = 255;   // A
    }

    ctx.putImageData(imageData, 0, 0);
}

function fillZone(x1: number, y1: number, x2: number, y2: number, value: number, xSize: number, ySize: number, space: number[]) {
    for (let i = 0; i < xSize; i++) {
        for (let j = 0; j < ySize; j++) {
            if (i >= x1 && i <= x2 && j >= y1 && j <= y2) {
                space[i + j * xSize] = value;
            }
        }
    }
    return space
}

const XSize = 500;
const YSize = 500;
const dt = 1 / 60;

    // Initialize GPU
var wave = new WaveSimulator();
let XSpace = getSpace(XSize, YSize, 0);

XSpace[XSize/2 + YSize * XSize / 2] = 300;

let VSpace = getSpace(XSize, YSize, 0);

CSpace = getSpace(XSize, YSize, 1);



console.log("GPU initialized");

async function main(): Promise<void> {

    await wave.loadOnGpu(
        XSize,
        YSize,
        dt,
        new Float32Array(XSpace),
        new Float32Array(VSpace),
        new Float32Array(CSpace)
    );
    canvas.width = XSize;
    canvas.height = YSize;

    // Initial render
    renderToCanvas(XSpace, XSize, YSize, CSpace);
    setInterval(async () => {
        // let now = Date.now();
        await wave.update();
        XSpace = Array.from(await wave.readXOnCpu());
        renderToCanvas(XSpace, XSize, YSize, CSpace);
    }, 1000 * dt); 
}

const canvas = document.getElementById('waveCanvas') as HTMLCanvasElement;

interface FillCRectEvent extends MouseEvent {}

const fillCRect: (event: FillCRectEvent) => Promise<void> = async (event) => {
    const rect: DOMRect = canvas.getBoundingClientRect();

    const px: number = event.clientX - rect.left;
    const py: number = event.clientY - rect.top;

    const gridX: number = Math.floor(px * XSize / rect.width);
    const gridY: number = Math.floor(py * YSize / rect.height);

    //const { X, V } = await wave.readOnCpu();
    //XSpace = Array.from(X);
    //VSpace = Array.from(V);

    CSpace = fillZone(
        gridX - 10, gridY - 10,
        gridX + 10, gridY + 10,
        0,      // value to set
        XSize, YSize,
        CSpace
    );

    wave.changeCSpace(new Float32Array(CSpace));
    console.log("CSpace changed");
};

canvas.addEventListener('click', fillCRect);
  

var mouseDown = false;

canvas.addEventListener('mousedown', async (event) => {
    mouseDown = true;
});

canvas.addEventListener('mouseup', async (event) => {
    mouseDown = false;
});

canvas.addEventListener('mousemove', async (event) => {
    if (mouseDown) {
        await fillCRect(event);
    }
});

    

main().catch(console.error);
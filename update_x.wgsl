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
}
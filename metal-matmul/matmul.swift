//
//  matmul.swift
//  metal-matmul
//
//  Created by Ankush Chatterjee on 10/11/24.
//

import Metal

struct MatrixDescriptor {
    var aRows: Int32
    var aCols: Int32
    var bRows: Int32
    var bCols: Int32
}

enum MatMulError : Error {
    case runtimeError(msg: String)
}

func encodeMatrixDescriptor(_ params: MatrixDescriptor) -> [UInt8] {
   var bytes: [UInt8] = []
   
   // Encode each Int as 4 bytes (32 bits)
   bytes.append(contentsOf: withUnsafeBytes(of: params.aRows.littleEndian) { Array($0) })
   bytes.append(contentsOf: withUnsafeBytes(of: params.aCols.littleEndian) { Array($0) })
   bytes.append(contentsOf: withUnsafeBytes(of: params.bRows.littleEndian) { Array($0) })
   bytes.append(contentsOf: withUnsafeBytes(of: params.bCols.littleEndian) { Array($0) })
   
   return bytes
}


func flatten(_ array: [[Float]]) -> [Float] {
    return array.flatMap { $0 }
}

func gpuMatMul(matA: [[Float]], matB: [[Float]]) throws -> [[Float]] {
    if (matA.count == 0 || matB.count == 0) {
        throw MatMulError.runtimeError(msg: "empty matrices")
    }
    if (matA[0].count != matB.count) {
        throw MatMulError.runtimeError(msg: "matA.cols != matB.rows")
    }
        
    // get gpu device
    let gpuDevice = MTLCreateSystemDefaultDevice()
   print("Using GPU device = \(gpuDevice!.name)")
    guard let library = gpuDevice?.makeDefaultLibrary()
    else { throw  MatMulError.runtimeError(msg: "makeDefaultLibrary") }

    let add_function = library.makeFunction(name: "matmul")
    
    // create pipeline state
    let additionComputePipelineState: MTLComputePipelineState!
    do {
        additionComputePipelineState = try gpuDevice?.makeComputePipelineState(function: add_function!)
    } catch {
        throw MatMulError.runtimeError(msg: "makeComputePipelineState")
    }
    guard let commandQueue = gpuDevice?.makeCommandQueue()
    else {throw MatMulError.runtimeError(msg: "makeCommandQueue")}

    guard let commandBuffer = commandQueue.makeCommandBuffer()
    else {throw MatMulError.runtimeError(msg: "makeCommandBuffer")}
    
    // make buffers
    let r1: Int32 = Int32(matA.count), c1: Int32 = Int32(matA[0].count)
    let r2: Int32 = c1, c2: Int32 = Int32(matB[0].count)
    let byteLen1 = Int(r1) * Int(c1) * MemoryLayout<Float>.size;
    let byteLen2 = Int(r2) * Int(c2) * MemoryLayout<Float>.size;
    let rByteLen = Int(r1) * Int(c2) * MemoryLayout<Float>.size;
    let params = MatrixDescriptor(aRows: r1, aCols: c1, bRows: r2, bCols: c2)
    let paramsBytes = encodeMatrixDescriptor(params)
    
    let buffMatA = gpuDevice?.makeBuffer(bytes: flatten(matA), length: byteLen1, options: .storageModeShared)
    let buffMatB = gpuDevice?.makeBuffer(bytes: flatten(matB), length: byteLen2, options: .storageModeShared)
    let buffParams = gpuDevice?.makeBuffer(bytes: paramsBytes, length: paramsBytes.count, options: .storageModeShared)
    
    let buffResult = gpuDevice?.makeBuffer(length: rByteLen, options: .storageModeShared)
    
    guard let commandEncoder = commandBuffer.makeComputeCommandEncoder()
    else {throw MatMulError.runtimeError(msg: "makeComputeCommandEncoder")}
    commandEncoder.setBuffer(buffMatA, offset: 0, index: 0)
    commandEncoder.setBuffer(buffMatB, offset: 0, index: 1)
    commandEncoder.setBuffer(buffResult, offset: 0, index: 2)
    commandEncoder.setBuffer(buffParams, offset: 0, index: 3)
    commandEncoder.setComputePipelineState(additionComputePipelineState)
    
    // dispatch threads, the size of threads per gird matches the size of resultant matrix
    let threadsPerGrid = MTLSize(width: Int(c2) , height: Int(r1), depth: 1)
    let w = additionComputePipelineState.threadExecutionWidth
    let h = additionComputePipelineState.maxTotalThreadsPerThreadgroup / w;
    let threadPerThreadGroup = MTLSize(width: w, height: h, depth: 1)
    
    commandEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadPerThreadGroup)
    commandEncoder.endEncoding()
    
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    var resultBufferPointer = buffResult?.contents().bindMemory(to: Float.self, capacity: rByteLen)
    
    var resultMat : [[Float]] = Array(repeating: Array(repeating: 0.0, count: Int(c2)), count: Int(r1))
    
    for i in  0..<Int(r1) {
        for j in 0..<Int(c2) {
            resultMat[i][j] = resultBufferPointer!.pointee
            resultBufferPointer = resultBufferPointer!.advanced(by: 1)
        }
    }
    return resultMat
}


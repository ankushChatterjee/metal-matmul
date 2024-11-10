//
//  main.swift
//  metal-matmul
//
//  Created by Ankush Chatterjee on 10/11/24.
//

import Foundation

// change these values for size
let r1 = 10, c1 = 2;
let r2 = c1, c2 = 4;

func generateRandomMatrix(rows: Int, cols: Int) -> [[Float]] {
    return (0..<rows).map { _ in
            (0..<cols).map { _ in
                Float.random(in: 0...256)
            }
        }
}

print("generating random arrays for: (\(r1),\(c1)) * (\(r2), \(c2))")
let matA = generateRandomMatrix(rows: r1, cols: c1);
let matB = generateRandomMatrix(rows: r2, cols: c2);

print("performaing matrix multiplication")
let result = try gpuMatMul(matA: matA, matB: matB)

if (r1 < 100 && c1 < 100 && c2 < 100) {
    print(result)
} else {
    print("matmul completed")
}


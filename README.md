# Structure-from-Motion
This project implements a Structure from Motion pipeline for obtaining monocular camera poses w.r.t the scene and reconstructing a 3D scene. 

Please refer to the [Report](Report.pdf) for implementation details:

## Algorithm Outline:
1. Feature Matching and Outlier rejection using RANSAC
2. Estimating Fundamental Matrix
3. Estimating Essential Matrix from Fundamental Matrix
4. Estimate Camera Pose from Essential Matrix
5. Check for Cheirality Condition using Triangulation
6. Perspective-n-Point
7. Bundle Adjustment

## Input Images:
A set of 5 calibrated images of Worcester Polytechnic Institute's Unity Hall:

![Imgs](https://github.com/miheer-diwan/Structure-from-Motion/assets/79761017/427fc910-9dd2-466b-a469-5cdca38d260f)

## SfM Output:
![All](https://github.com/miheer-diwan/Structure-from-Motion/assets/79761017/e31555f7-764d-440c-9c50-f35039887e2c)

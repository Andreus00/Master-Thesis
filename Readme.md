
# Todo

## Pipeline Enhancement:
- [ ] Add a mesh extraction step to the pipeline (code from Human 3Diffusion)
- [ ] Improve code to avoid knn after avatar canonization
- [ ] Improve code to avoid knn after garment retargeting
- [ ] Add a step to register SMPL during evaluation


## Data:    
- [x] Download examples of many different people. Each example should have:
    - [x] Mesh path
    - [x] Clothes labels
    - [x] SMPL registration
    - [x] Multiview render of each single garment
    - [x] Multiview render of the person with the garment
    - [x] Mesh of clothes


## Metrics:
2D perception mertrics:
- [x] SSIM
- [x] PSNR
- [ ] LPIPS
3D metrics:
- [ ] Chamfer distance
- [ ] Normal consistency

## Experiments:
We need to run the following experiments:
- [ ] Quantitative evaluation of the 3D reconstruction vs HaveFun on new data
- [ ] Quantitative evaluation of the full pipeline vs HaveFun + GALA
- [ ] Quantitative evaluation of the full pipeline vs 2D vton

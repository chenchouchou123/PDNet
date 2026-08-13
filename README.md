# PDNet
PDNet: Physics-Decoupling Network for mmWave 3D Human Pose Estimation



## Gallery



https://github.com/user-attachments/assets/8af3c076-fc90-42ba-bdd6-f1b55f861b2c



https://github.com/user-attachments/assets/929c35e4-8b60-41df-982c-34d80aa71180



https://github.com/user-attachments/assets/dc810403-19c8-4549-9e7e-2cf22dc1e821


https://github.com/user-attachments/assets/ac36429b-4813-4370-8eae-18a73118899f



## Abstract

Millimeter-wave (mmWave) radar has emerged as a highly promising, privacy-preserving sensor for 3D human pose estimation (HPE). However, the primary challenge lies in excavating fine-grained, pose-relevant features from complex and noisy radar echoes. Existing deep learning methods typically treat radar spectrograms as generic pixel arrays. This purely data-driven paradigm fundamentally neglects the underlying electromagnetic physics, leaving models highly susceptible to clutter and incapable of effectively disentangling macroscopic torso translations from microscopic limb kinematics. To overcome these bottlenecks, we propose Physics-Decoupling Network (PDNet). We shift mmWave HPE from data-driven texture learning to physics-guided representation learning via explicit decoupling of phase and Doppler polarity. Specifically, extracting pose-related physical priors is exceptionally difficult due to the inherent aliasing of amplitude and phase, as well as the neutralization of concurrent bidirectional limb movements. To tackle this, PDNet designs two explicit front-ends: a Phase Decoupling Module (PDM) that employs multi-scale complex rotary modulation to decouple the macro-gait phase from micro-burst amplitude in the complex feature space; and a Sign Decoupling Module (SDM) that utilizes a multi-head routing mechanism to dynamically disentangle overlapping bidirectional micro-Doppler signals into independent static, approaching, and receding kinematic branches. These decoupled physical attributes are then aggregated into explicit global priors and injected into the visual backbone via Token Prompting Module (TPM), endowing the deep representations with strict top-down kinematic constraints. Extensive experiments demonstrate that PDNet outperforms state-of-the-art approaches, reducing the pose estimation error by 11.1\% and 7.6\% on the MVDoppler-Pose and HuPR datasets.



### Framework

[Figure_3.pdf](https://github.com/user-attachments/files/31036859/Figure_3.pdf)


## Code
Environment:
Python: 3.10.8
Pytorch: 1.13.1
CUDA: 11.6
CuDNN: 8
Environment can directly be imported through:[MVDoppler-pose](https://github.com/gogoho88/MVDoppler-Pose).


Training:
Edit the corresponding path and variables in the 'conf' folder.
PDNet training:
```
docker run --gpus all --shm-size=60g -d -v /your path -p 8080:8080 --name mmwave gogoho88/stanford_mmwave:v3 python /workspace/main_train.py
```
PDNet testing:
```
docker run --gpus all --shm-size=60g -d -v /your path -p 8080:8080 --name mmwave gogoho88/stanford_mmwave:v3 python /workspace/main_inference.py
```

## Related Projects

Our code is based on [MVDoppler-pose](https://github.com/gogoho88/MVDoppler-Pose).

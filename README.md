
# PDNet
PDNet: Physics-Decoupling Network for mmWave 3D Human Pose Estimation





## Gallery


<table align="center">
  <tr>
    <td align="center">texting<br>
      <img width="450" height="150" alt="20220614100950-2-texting-PoseVideo-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/0950ea69-4419-4b28-829c-7b14b1073578" />
    </td>
    <td align="center">normal<br>
      <img width="450" height="150" alt="20220729134757-5-normal-PoseVideo-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/4f0dadc6-f9b8-488c-943a-751809cf0616" />
    </td>
  </tr>
  <tr>
    <td align="center">pockets<br>
      <img width="450" height="150" alt="20220614103946-2-pockets-PoseVideo-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/3a8d01da-222f-4df7-8588-a7099f070d32" />
    </td>
    <td align="center">phone_call<br>
      <img width="450" height="150" alt="20220614102503-2-phone_call-PoseVideo-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/c2af66c5-58c9-4c6e-81aa-57bc09bb1a5a" />
    </td>
  </tr>
</table>





## Abstract

Millimeter-wave (mmWave) radar has emerged as a highly promising, privacy-preserving sensor for 3D human pose estimation (HPE). However, the primary challenge lies in excavating fine-grained, pose-relevant features from complex and noisy radar echoes. Existing deep learning methods typically treat radar spectrograms as generic pixel arrays. This purely data-driven paradigm fundamentally neglects the underlying electromagnetic physics, leaving models highly susceptible to clutter and incapable of effectively disentangling macroscopic torso translations from microscopic limb kinematics. To overcome these bottlenecks, we propose Physics-Decoupling Network (PDNet). We shift mmWave HPE from data-driven texture learning to physics-guided representation learning via explicit decoupling of phase and Doppler polarity. Specifically, extracting pose-related physical priors is exceptionally difficult due to the inherent aliasing of amplitude and phase, as well as the neutralization of concurrent bidirectional limb movements. To tackle this, PDNet designs two explicit front-ends: a Phase Decoupling Module (PDM) that employs multi-scale complex rotary modulation to decouple the macro-gait phase from micro-burst amplitude in the complex feature space; and a Sign Decoupling Module (SDM) that utilizes a multi-head routing mechanism to dynamically disentangle overlapping bidirectional micro-Doppler signals into independent static, approaching, and receding kinematic branches. These decoupled physical attributes are then aggregated into explicit global priors and injected into the visual backbone via Token Prompting Module (TPM), endowing the deep representations with strict top-down kinematic constraints. Extensive experiments demonstrate that PDNet outperforms state-of-the-art approaches, reducing the pose estimation error by 11.1\% and 7.6\% on the MVDoppler-Pose and HuPR datasets.



### Framework

![Framework Description]<img width="1454" height="1267" alt="Figure_3" src="https://github.com/user-attachments/assets/96c9e16d-718e-42be-a6f5-f4c13a55fa84" />



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

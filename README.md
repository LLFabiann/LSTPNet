# LSTPNet
Official code repository for paper "LSTPNet: Long Short-Term Perception Network for Dynamic Facial Expression Recognition in the Wild".

## Overall Architecture
<p align="center">
    <img src="./img/LSTPNet.png" width="100%"/> <br />
 <em> 
    Overall architecture of the proposed LSTPNet.
    </em>
</p>

## Test
Please download pretrained_weights at [Google Driver](https://drive.google.com/drive/folders/1_nqZ21ZSd0RXY4f4auLt-1Qn4G4tIp4S?usp=drive_link).

Please run `python test_DFEW.py`, `python test_FERV39k.py`, `python test_AFEW.py`, respectively.

## Results
<p align="center">
    <img src="img/DFEW_result.png" width="100%"/> <br />
 <em> 
    Comparison with state-of-the-art methods on DFEW. The best results are highlighted in bold.
    </em>
</p>

<p align="center">
    <img src="img/FERV39k_result.png" width="50%"/> <br />
 <em> 
    Comparison with state-of-the-art methods on FERV39k. The best results are highlighted in bold.
    </em>
</p>

<p align="center">
    <img src="img/AFEW_result.png" width="50%"/> <br />
 <em> 
    Comparison with state-of-the-art methods on AFEW. The best results are highlighted in bold.
    </em>
</p>

## Citation
Please cite the following paper if you use this repository in your research:
```
@article{LU2024104915,
    title = {LSTPNet: Long short-term perception network for dynamic facial expression recognition in the wild},
    author = {Chengcheng Lu and Yiben Jiang and Keren Fu and Qijun Zhao and Hongyu Yang},
    journal = {Image and Vision Computing},
    volume = {142},
    pages = {104915},
    year = {2024}
}
```

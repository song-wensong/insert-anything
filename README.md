<h1 align="center">Insert Anything</h2>
<p align="center">
<a href="https://song-wensong.github.io/"><strong>Wensong Song</strong></a>
·
<a href="https://openreview.net/profile?id=~Hong_Jiang4"><strong>Hong Jinag</strong></a>
·
<a href="https://z-x-yang.github.io/"><strong>Zongxing Yang</strong></a>
·
<a href="https://scholar.google.com/citations?user=WKLRPsAAAAAJ&hl=en"><strong>Ruijie Quan</strong></a>
·
<a href="https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=en"><strong>Yi Yang</strong></a>
<br>
<br>
    <a href="https://arxiv.org/pdf/2504.15009"><img src='https://img.shields.io/badge/arXiv-InsertAnything-red?color=%23aa1a1a' alt='Paper PDF'></a>
    <a href='https://song-wensong.github.io/insert-anything/'><img src='https://img.shields.io/badge/Project%20Page-InsertAnything-cyan?logoColor=%23FFD21E&color=%23cbe6f2' alt='Project Page'></a>
    <a href='https://huggingface.co/spaces/xichenhku/AnyDoor-online'><img src='https://img.shields.io/badge/Hugging%20Face-InsertAnything-yellow?logoColor=%23FFD21E&color=%23ffcc1c'></a>
<br>
<b>Zhejiang University &nbsp; | &nbsp; Harvard University &nbsp; | &nbsp;  Nanyang Technological University </b>
</p>

## 🔥 News

* **[Soon]** Release train code.
* **[Soon]** Release **AnyInsertion** dataset on HuggingFace.
* **[Soon]** Support online demo on HuggingFace.
* **[2025.4.22]** Release inference demo and pretrained checkpoint.


## 🛠️ Installation

Begin by cloning the repository:

```bash
git clone https://github.com/song-wensong/insert-anything
cd insert-anything
```

### Installation Guide for Linux

Conda's installation instructions are available [here](https://docs.anaconda.com/free/miniconda/index.html).

```shell
conda create -n insertanything python==3.10

conda activate insertanything

pip install -r requirements.txt
```


## ⏬ Download Checkpoints
*   **Insert Anything Model:** Download the main checkpoint from [HuggingFace](https://huggingface.co/WensongSong/Insert-Anything) and replace `/path/to/lora` in inference.py.
*   **FLUX.1-Fill-dev Model:** This project relies on [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) and [FLUX.1-Redux-dev ](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) as components. Download its checkpoint(s) as well and replace `/path/to/black-forest-labs-FLUX.1-Fill-dev` and `/path/to/black-forest-labs-FLUX.1-Redux-dev`.

## 🎥 Inference
### Using Command Line
```bash
cd insert-anything

python inference.py
```


## 💡 Tips

To run mask-prompt examples, you may need to obtain the corresponding masks. You can choose to use [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) or the draw_mask script provided by us

```Bash
python draw_mask.py 
```



## 🤝 Acknowledgement

We appreciate the open source of the following projects:

* [diffusers](https://github.com/huggingface/diffusers)
* [OminiControl](https://github.com/Yuanshi9815/OminiControl)

## Citation
```
@misc{song2025insertanythingimageinsertion,
      title={Insert Anything: Image Insertion via In-Context Editing in DiT}, 
      author={Wensong Song and Hong Jiang and Zongxing Yang and Ruijie Quan and Yi Yang},
      year={2025},
      eprint={2504.15009},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.15009}, 
}
```
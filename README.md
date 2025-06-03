<h1 align="center">Insert Anything</h2>
<p align="center">
<a href="https://song-wensong.github.io/"><strong>Wensong Song</strong></a>
·
<a href="https://openreview.net/profile?id=~Hong_Jiang4"><strong>Hong Jiang</strong></a>
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
    <a href='https://huggingface.co/spaces/WensongSong/Insert-Anything'><img src='https://img.shields.io/badge/Hugging%20Face-InsertAnything-yellow?logoColor=%23FFD21E&color=%23ffcc1c'></a>
<br>
<b>Zhejiang University &nbsp; | &nbsp; Harvard University &nbsp; | &nbsp;  Nanyang Technological University </b>
</p>

## 🔥 News

* **[2025.6.3]** Separate the ComfyUI code into a [new repository](https://github.com/mo230761/InsertAnything-ComfyUI-official).
* **[2025.6.1]** Release a new ComfyUI workflow! No need to download the full model folder!
* **[2025.5.23]** Release the training code for users to reproduce results and adapt the pipeline to new tasks!
* **[2025.5.13]** Release **AnyInsertion** text-prompt dataset on [HuggingFace](https://huggingface.co/datasets/WensongSong/AnyInsertion_V1).
* **[2025.5.9]** Release demo video of the Hugging Face Space, now available on [YouTube](https://www.youtube.com/watch?v=IbVcOqXkyXo) and [Bilibili]( https://www.bilibili.com/video/BV1uX55z5EtN/?share_source=copy_web&vd_source=306bd420c358f5d468394a1eb0f7b1ad).
* **[2025.5.7]** Release inference for nunchaku demo to support **10GB VRAM**.
* **[2025.5.6]** Support ComfyUI integration for easier workflow management.
* **[2025.5.6]** Update inference demo to support **26GB VRAM**, with increased inference time.
* **[2025.4.26]** Support online demo on [HuggingFace](https://huggingface.co/spaces/WensongSong/Insert-Anything).
* **[2025.4.25]** Release **AnyInsertion** mask-prompt dataset on [HuggingFace](https://huggingface.co/datasets/WensongSong/AnyInsertion_V1).
* **[2025.4.22]** Release inference demo and pretrained [checkpoint]((https://huggingface.co/WensongSong/Insert-Anything)).


## 💡 Demo

![Insert Anything Teaser](docs/InsertAnything_files/images/Insert-Anything-Github-teaser_00.png)

For more demos and detailed examples, check out our project page: <a href="https://song-wensong.github.io/insert-anything/"><img src="https://img.shields.io/badge/Project%20Page-InsertAnything-cyan?logoColor=%23FFD21E&color=%23cbe6f2" alt="Project Page" /></a>

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

**10 VRAM :**
*   **Insert Anything Model:** Download the main checkpoint from [HuggingFace](https://huggingface.co/aha2023/insert-anything-lora-for-nunchaku) and replace `/path/to/lora-for-nunchaku` in inference_for_nunchaku.py.

*   **FLUX.1-Fill-dev Model:** This project relies on [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) and [FLUX.1-Redux-dev ](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) as components. Download its checkpoint(s) as well and replace `/path/to/black-forest-labs-FLUX.1-Fill-dev` and `/path/to/black-forest-labs-FLUX.1-Redux-dev`.

*   **Nunchaku-FLUX.1-Fill-dev Model:** Download the main checkpoint from [HuggingFace](https://huggingface.co/mit-han-lab/svdq-int4-flux.1-fill-dev) and replace `/path/to/svdq-int4-flux.1-fill-dev`.


**26 or 40 VRAM :**
*   **Insert Anything Model:** Download the main checkpoint from [HuggingFace](https://huggingface.co/WensongSong/Insert-Anything) and replace `/path/to/lora` in inference.py and app.py.

*   **FLUX.1-Fill-dev Model:** This project relies on [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) and [FLUX.1-Redux-dev ](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) as components. Download its checkpoint(s) as well and replace `/path/to/black-forest-labs-FLUX.1-Fill-dev` and `/path/to/black-forest-labs-FLUX.1-Redux-dev`.




## 🎥 Inference
### 10 VRAM
We are very grateful to @[judian17](https://github.com/judian17) for providing the nunchaku version of LoRA.After downloading the required weights, you need to go to the official [nunchaku repository](https://github.com/mit-han-lab/nunchaku) to install the appropriate version of nunchaku.
```bash
python inference_for_nunchaku.py
```



### 26 or 40 VRAM
```bash
python inference.py
```


## 🖥️ Gradio
### Using Command Line
```bash
python app.py
```


## 🧩 ComfyUI

We have specially created a [repository](https://github.com/mo230761/InsertAnything-ComfyUI-official) for the workflow and you can check the repository and have a try!

## 🧩 ComfyUI in community
We deeply appreciate the community of developers who have created innovative applications based on the Insert Anything model. Throughout this development process, we have received invaluable feedback. As we continue to enhance our models, we will carefully consider these insights to further optimize our models and provide users with a better experience.

Below is a selection of community‑created workflows along with their corresponding tutorials:

<table>
<tbody>
  <tr>
    <td>Workflow</td>
    <td>Author</td>
    <td>Tutorial</td>
  </tr>
  <tr>
    <td><a href="https://openart.ai/workflows/t8star/insert-anything/hvMbsN7LXEAoMGZgv1fL"> Insert Anything极速万物迁移图像编辑优化自动版 </a> </td>
    <td><a href="https://openart.ai/workflows/profile/t8star?sort=latest"> T8star-Aix </a></td>
    <td>
        <a href="https://youtu.be/rV4tw1PKh-4?si=XDK5O-SCT51aF5fW">
          YouTube
        </a>|
        <a href="https://www.bilibili.com/video/BV1qjE7zgEhe/?spm_id_from=333.337.search-card.all.click&vd_source=6430895fb8ccabaed9a88151abe4a723">
          Bilibili
        </a>
    </td>
  </tr>
</tbody>
</table>

## 💡 Tips

🔷  To run mask-prompt examples, you may need to obtain the corresponding masks. You can choose to use [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) or the draw_mask script provided by us

```Bash
python draw_mask.py 
```

🔷  The mask must fully cover the area to be edited.



## ⏬ Download Dataset
*   **AnyInsertion dataset:** Download the AnyInsertion dataset from [HuggingFace](https://huggingface.co/datasets/WensongSong/AnyInsertion_V1).

## 🚀 Training

### 🔷  Mask-prompt Training

*   **Replace flux model paths:** Replace /path/to/black-forest-labs-FLUX.1-Fill-dev and /path/to/black-forest-labs-FLUX.1-Redux-dev in experiments/config/insertanything.yaml


*   **Download mask-prompt dataset:** Download the AnyInsertion mask-prompt dataset from [HuggingFace](https://huggingface.co/datasets/WensongSong/AnyInsertion_V1).


*   **Convert parquet to image:** Use the script `parquet_to_image.py` to convert Parquet files to images.

*   **Test(Optional):** If you want to perform testing during the training process, you can modify the test path under the specified file `src/train/callbacks.py`(line 350). The default does not require a testing process.

*   **Run the training code:** Follow the instruction :
   
    ```Bash
    bash scripts/train.sh
    ```


## 🤝 Acknowledgement

We appreciate the open source of the following projects:

* [diffusers](https://github.com/huggingface/diffusers)
* [OminiControl](https://github.com/Yuanshi9815/OminiControl)

## Citation
```
@article{song2025insert,
  title={Insert Anything: Image Insertion via In-Context Editing in DiT},
  author={Song, Wensong and Jiang, Hong and Yang, Zongxing and Quan, Ruijie and Yang, Yi},
  journal={arXiv preprint arXiv:2504.15009},
  year={2025}
}
```

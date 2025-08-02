# <center> AirRoom: Objects Matter in Room Reidentification

This is the official implementation of the following publication:

> **AirRoom: Objects Matter in Room Reidentification**  
> [Runmao Yao](https://21yrm.github.io/), [Yi Du](https://sairlab.org/yid/), [Zhuoqun Chen](https://sairlab.org/zhuoqunc/), [Haoze Zheng](https://sairlab.org/haozez/), [Chen Wang](https://sairlab.org/chenw/)  
> *CVPR 2025*  
> **[arXiv](https://arxiv.org/abs/2503.01130)** | **[Project Page](https://sairlab.org/airroom/)**

## Requirements

The code has been tested on:

- CUDA 11.5
- GeForce RTX 3090 Ti (24GB).

## Setup

1. **Clone the repository and its submodules**

    ```
    git clone https://github.com/21yrm/AirRoom.git
    cd AirRoom
    ```

2. **Build the Docker container**

    ```
    docker login
    ./build_docker.sh
    ```

3. **Download the dataset**

    Download the room re-identification dataset from the following link: [Google Drive – Room Re-identification Dataset](https://drive.google.com/drive/folders/1Qc-53BXzKZDBVpaVtpaep-37zQpfQDcJ)

    After downloading, unzip the `datasets.zip` file and place the extracted folder under the project root directory.

    The expected dataset directory structure should look like:

    ```
    datasets/
    ├── <dataset_1>/
    │   ├── <scene_1>/
    │   │   ├── <room_1>/
    │   │   │   ├── rgb/
    │   │   │   └── depth/
    │   │   └── ...
    │   ├── ...
    │   └── room_label.txt
    └── ...
    ```

## How to Run

Follow the steps below to run the project:

1. **Launch the Docker Container**

    ```
    ./run_docker.sh
    ```

2. **Preprocess the Dataset (Construct Reference Database)**

    - Open `config/preprocess.yaml`

    - Set the field `dataset_name` to the desired dataset (e.g., `MPReID`)

    - Then run:

        ```
        python preprocess.py
        ```

3. **Fix Compatibility Issues for Python 3.8 and PyTorch 1.x**

    Some operations are not supported in Python 3.8 and PyTorch 1.x. Please modify the following files accordingly:
    
    `cache/torchhub/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py` 
    
    - **Line 58:**
        ```
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
        ```
    - Change to:
        ```
        self, init_attn_std = None, init_proj_std = None, factor: float = 1.0
        ```
    
    - **Before Line 69:**

        Insert the following function before the `forward` method:

        ```
        def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
            is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
            import math
            L, S = query.size(-2), key.size(-2)
            scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
            attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
            if is_causal:
                assert attn_mask is None
                temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                attn_bias.to(query.dtype)

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
                else:
                    attn_bias = attn_mask + attn_bias

            if enable_gqa:
                key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
                value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

            attn_weight = query @ key.transpose(-2, -1) * scale_factor
            attn_weight += attn_bias
            attn_weight = torch.softmax(attn_weight, dim=-1)
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            return attn_weight @ value
        ```
    
    - **Lines 102–104:**
        ```
        x = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        ```
    - Change to:
        ```
        x = self.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        ```
    
    `cache/torchhub/hub/facebookresearch_dinov2_main/dinov2/layers/block.py`

    - **Lines 150–152:**
        ```
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        ```
    - Change to:
        ```
        init_attn_std = None,
        init_proj_std = None,
        init_fc_std = None,
        ```
    
    Then re-run:

    ```
    python preprocess.py
    ```

4. **Run Inference**

    - Modify `config/inference.yaml` to specify the target dataset and desired configuration.

    - Then execute:

        ```
        python inference.py
        ```


## Citing AirRoom

If you find our work interesting, please consider citing us!

```
@InProceedings{Yao_2025_CVPR,
    author    = {Yao, Runmao and Du, Yi and Chen, Zhuoqun and Zheng, Haoze and Wang, Chen},
    title     = {AirRoom: Objects Matter in Room Reidentification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {1385-1394}
}
```

## Related Projects

We gratefully acknowledge the following open-source projects that significantly contributed to our work:

- **Semantic-SAM**, for its outstanding instance segmentation performance.

- **LightGlue**, for its excellent local feature matching capabilities.

# Hickathon 6

## Installation

1.  Install `uv`:

    ```shell
    pip install uv
    ```

2.  Create a virtual environment:

    ```shell
    uv venv
    ```

3.  Activate the virtual environment:

    -   On Windows:

        ```shell
        .venv\Scripts\activate
        ```

    -   On macOS/Linux:

        ```shell
        source .venv/bin/activate
        ```

4.  Install the project in editable mode:

    ```shell
    uv pip install -e .
    ```

## Exact reproduction
In the `data` folder, there should be two files we created for the preprocessing.

In order to reproduce our results do the following steps : 
 - in `data` folder, add `X_train.csv`, `X_test.csv`, `y_train.csv`
 - Run `src/hickathon_six/preprocessing.ipynb, this should generate the `data/df_processed.csv` file.
 - For the DAE embedding generation, one way is to reproduce our results is to retrain the DAE models from scratch. To do that:
    - In the `src/hickaton_six` folder, edit the `dae_exo.py` and `dae_que.py` files to setup WandB loging and artifact storing (see `src/hickaton_six/DAE/config.py`). To train, the models directly (without k_fold) and get the embeddings, run the `rerun_global_finetune_exo.py` and `rerun_global_finetune_que.py` file. If you want to run the k_folds too, run the `dae_exo.py` and `dae_que.py` files. For all this files, a gpu instance and torch with cuda support is needed.
    - **OR** if you trust us with the training and finetunning of the DAEs, the artifacts of our runs are available publicly (the whole wandb projet is public and accessible at https://wandb.ai/themlaw-personal/hi6/). You should be able to generate the embeddings by running the `dae_que_generate.py` and `dae_exo_generate.py` files (a gpu is also needed)
 - Put in `data/Embedding_wandb` folder the embeddings csv files named as `[exo|que]_embedding_[dae|finetuned]_full_[train|test].csv`
 - Finally run the `autogluon.ipynb` notebook (a gpu is suggested to get the same result as us, but you can also change the configuration of the regressor with a less compute intensive approach)

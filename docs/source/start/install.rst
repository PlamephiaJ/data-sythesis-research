Installation
============

Requirements
------------

- **Python**: Version >= 3.9
- **CUDA**: Version >= 12.4

Install from Docker Image
-------------------------

This feature is coming soon. Stay tuned for updates on Docker image availability.

Install from Custom Environment
-------------------------------

We recommend using Docker images for convenience. However, if your environment is not compatible with the Docker image, you can install `smf` in a Python environment.

Pre-requisites
==============

To ensure a smooth installation of `smf`, you need to set up the necessary CUDA and cuDNN dependencies. These are critical for leveraging GPU acceleration in `smf`'s operations.

We need to install the following pre-requisites:

- **CUDA**: Version >= 12.4
- **cuDNN**: Version >= 9.8.0

CUDA version 12.4 or higher is recommended to align with the Docker image. For other CUDA versions, refer to `NVIDIA CUDA website <https://developer.nvidia.com/cuda-toolkit-archive>`_.

.. code:: bash

    # Change directory to a location of your choice; avoid the smf source code directory
    wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
    dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
    cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    apt-get -y install cuda-toolkit-12-4
    update-alternatives --set cuda /usr/local/cuda-12.4

cuDNN can be installed using the following commands. For other cuDNN versions, refer to `NVIDIA cuDNN website <https://developer.nvidia.com/rdp/cudnn-archive>`_.

.. code:: bash

    # Change directory to a location of your choice; avoid the smf source code directory
    wget https://developer.download.nvidia.com/compute/cudnn/9.8.0/local_installers/cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
    dpkg -i cudnn-local-repo-ubuntu2204-9.8.0_1.0-1_amd64.deb
    cp /var/cudnn-local-repo-ubuntu2204-9.8.0/cudnn-*-keyring.gpg /usr/share/keyrings/
    apt-get update
    apt-get -y install cudnn-cuda-12

Install Dependencies
====================

.. note::

    We recommend using a fresh Conda environment to install `smf` and its dependencies to avoid conflicts.

    **Warning**: Inference frameworks like vLLM may override your installed PyTorch version if not carefully managed. To prevent this, install inference frameworks first with their required PyTorch version. If you want to use an existing PyTorch installation with vLLM, follow their official instructions at `Use an existing PyTorch installation <https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#build-wheel-from-source>`_.

1. First, set up a Conda environment with Python 3.10:

.. code:: bash

   conda create -n smf_env python=3.10
   conda activate smf_env

2. Next, execute the ``install.sh`` script provided in the `smf` repository to install dependencies:

.. code:: bash

    # Ensure the smf_env Conda environment is activated
    bash install.sh

If you encounter errors during this step, inspect the ``install.sh`` script and manually execute the steps outlined within it.

Install smf
------------

To install the latest version of `smf`, clone the repository and install it from source. This approach allows you to modify the code to customize post-training jobs.

.. code:: bash

   git clone https://rnd-gitlab-ca-g.huawei.com/vanpub/ai_firewall.git
   cd ai_firewall
   pip install --no-deps -e .

Post-installation
=================

After installation, verify that the installed packages have not been overridden by other installations. Pay particular attention to the following packages:

- **torch** and related torch packages
- **nvidia-cudnn-cu12**: Required for the Magetron backend

To check the installed versions, you can use:

.. code:: bash

   pip list | grep -E 'torch|nvidia-cudnn-cu12'

If any discrepancies are found, reinstall the affected packages to ensure compatibility with `smf`.

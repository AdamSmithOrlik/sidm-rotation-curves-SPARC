# sidm-rotation-curves-SPARC

A repository for EXPLORE V project investigating observed rotation curves from the SPARC dataset using the squashed Jeans model.

Instructions on how to set up a virtual environment:

1. Ensure that python is already installed on your machine

   - `which python` should show what version is installed

2. Inside of the GitHub repository open a terminal and create a virtual environment

```bash
python3 -m venv .sparcenv
```

Note: you can name it whatever you want e.g. '.venv' works fine too

3. Activate the environment

```bash
source .sparcenv/bin/activate
```

4. Install the jeans package into the virtual environment

```bash
pip install git+https://github.com/dark-physics/jeans.git
```

Once complete check out `getting-started.ipynb`. The code therein should now work.

# cookie_cutter_project

learning cookie cutter

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


## How to run

This project uses [uv](https://docs.astral.sh/uv/) for package management.

### 1. Preprocess data

First, process the raw data into the format needed for training:

```bash
uv run python src/cookie_cutter_project/data.py data/raw data/processed
```

### 2. Train the model

Train the CNN model on the processed MNIST data:

```bash
uv run python src/cookie_cutter_project/train.py
```

Optional arguments:
- `--lr`: Learning rate (default: 0.001)
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 10)

Example with custom parameters:
```bash
uv run python src/cookie_cutter_project/train.py --lr 0.0005 --batch-size 64 --epochs 20
```

### 3. Evaluate the model

Evaluate the trained model on the test set:

```bash
uv run python src/cookie_cutter_project/evaluate.py models/model.pth
```

### 4. Visualize embeddings

Generate a t-SNE visualization of the model's learned embeddings:

```bash
uv run python src/cookie_cutter_project/visualize.py models/model.pth
```

The visualization will be saved to `reports/figures/embeddings.png`.

---

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
# trigger CI

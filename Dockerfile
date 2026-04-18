FROM python:3.12-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
COPY robin/ robin/
COPY robin_demo.ipynb .
COPY robin_full.ipynb .

RUN SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 uv pip install --system -e '.[dev]'

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]

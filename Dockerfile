FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Pins mirror canonical-env.txt so sandbox computations agree with verification.
RUN python -m pip install \
    numpy==2.4.6 \
    pandas==3.0.5 \
    pyarrow==23.0.1 \
    scikit-learn==1.9.0 \
    scipy==1.17.1 \
    pyyaml==6.0.3 \
    joblib==1.5.3 \
    && mkdir /submission

WORKDIR /workspace

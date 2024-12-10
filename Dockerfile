FROM ghcr.io/osgeo/gdal:ubuntu-small-latest
RUN apt-get update && \
    apt-get install -y python3-pip python3-venv software-properties-common && \
    add-apt-repository ppa:git-core/ppa && \
    apt-get -y install git
WORKDIR /app
COPY ./app /app
RUN python3 -m venv /app/.venv && \
    /app/.venv/bin/pip install --upgrade pip && \
    /app/.venv/bin/pip install -r /app/requirements.txt
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8501
CMD ["/.venv/bin/streamlit", "run", "app.py"]
#https://github.com/OSGeo/gdal/pkgs/container/gdal
FROM ghcr.io/osgeo/gdal:ubuntu-small-latest

RUN apt-get update && \
    apt-get install -y python3-pip software-properties-common && \
    add-apt-repository ppa:git-core/ppa && \
    apt-get -y install git

COPY ./app .
RUN python -m pip install --break-system-packages -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py"]
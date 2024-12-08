FROM jupyter/pyspark-notebook

ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

WORKDIR /src

COPY prediction.py /src
COPY logistic_regression /src/logistic_regression

RUN pip install flask flask-cors numpy pyspark

EXPOSE 5000

CMD ["python", "prediction.py"]
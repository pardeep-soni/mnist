FROM tensorflow/tensorflow

COPY mnist.h5 /app/mnist_model.h5

WORKDIR /app

RUN pip install flask

COPY app.py /app/app.py

EXPOSE 5000

CMD ["python", "app.py"]

FROM python:3.13-slim
COPY . .
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt 
CMD ["python3", "app.py"]

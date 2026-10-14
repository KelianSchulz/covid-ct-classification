FROM python:3.11-slim

WORKDIR /app


RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN python -c "from torchvision import models; models.resnet18(weights=models.ResNet18_Weights.DEFAULT)"

COPY *.py ./

ENV DATA_DIR=/data
ENV OUTPUT_DIR=/app/output
ENV MPLBACKEND=Agg

CMD ["python", "evaluate_model.py"]
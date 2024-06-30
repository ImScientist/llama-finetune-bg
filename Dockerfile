FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/workspace

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]

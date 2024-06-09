FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ARG WORKDIR=/home
WORKDIR $WORKDIR

RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy --dev

# setting up jupyter notebook
RUN jupyter contrib nbextension install --system && \
    jt -t oceans16 -cellw 100% -lineh 170

COPY ["src", "./src"]

EXPOSE 8888

ENV PYTHONPATH=$WORKDIR

CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]

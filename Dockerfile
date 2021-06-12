FROM python:slim

RUN adduser --disabled-password vae

USER vae

WORKDIR /home/vae

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY *.py *.ipynb ./

EXPOSE 8888

ENTRYPOINT ["/home/vae/.local/bin/jupyter", "lab"]
CMD ["--ip=0.0.0.0"]

FROM python:slim

WORKDIR /home/saundersp/Variational_Auto_Encoder

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8888

CMD ["/usr/local/bin/jupyter", "lab"]

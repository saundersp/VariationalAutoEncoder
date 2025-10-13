FROM debian:13.1-slim

RUN apt-get update \
	&& apt-get install -y --no-install-recommends python3.13-venv=3.13.5-2 \
	&& ln -sv /usr/bin/python3.13 /usr/bin/python \
	&& apt-get autoremove -y \
	&& apt-get autoclean -y \
	&& rm -rv /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000

RUN groupadd -g ${GID} vae \
	&& useradd -m -l -g ${GID} -u ${UID} -G sudo,vae vae

WORKDIR /home/vae

USER vae

COPY helper.sh requirements.txt ./

RUN ./helper.sh i

COPY *.py *.ipynb ./

EXPOSE 8888

ENTRYPOINT ["./helper.sh"]
CMD ["l"]

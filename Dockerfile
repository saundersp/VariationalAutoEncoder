FROM nvidia/cuda:13.0.0-base-ubi9

RUN dnf install -y python3.12-3.12.9-1.el9_6.1 \
	&& dnf clean all \
	&& ln -s /usr/bin/python3.12 /usr/bin/python

ARG UID=1000
ARG GID=1000

# $(ls -la /dev/nvidia0 | cut -d ' ' -f 4)
ENV NVIDIA_GUID=27
RUN groupadd -g ${NVIDIA_GUID} vglusers \
	&& groupadd -g ${GID} vae \
	&& useradd -m -l -g ${GID} -u ${UID} -G vglusers,vae vae

WORKDIR /home/vae

USER vae

COPY helper.sh requirements.txt ./

RUN ./helper.sh i

COPY *.py *.ipynb ./

EXPOSE 8888

ENTRYPOINT ["./helper.sh"]
CMD ["l"]

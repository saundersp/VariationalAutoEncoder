FROM nvidia/cuda:12.5.1-devel-ubi9

RUN dnf install -y python3.12-3.12.1-4.el9 libcudnn8-8.9.7.29-1.cuda12.2 \
	&& dnf clean all \
	&& ln -s /usr/bin/python3 /usr/bin/python

ARG UID=1000
ARG GID=1000

#RUN groupadd -g "$(stat -c %g /dev/nvidia0)" vglusers && useradd -l -g ${GID} -u ${UID} -G vglusers vae
RUN groupadd -g 27 vglusers && groupadd -g ${GID} vae && useradd -l -g ${GID} -u ${UID} -G vglusers,vae vae

WORKDIR /home/vae

USER vae

COPY helper.sh requirements.txt ./

RUN ./helper.sh i

COPY *.py *.ipynb ./

EXPOSE 8888

ENTRYPOINT ["./helper.sh"]
CMD ["l"]

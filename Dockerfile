FROM python:3.9.6

WORKDIR /home

RUN apt update && apt upgrade -y && pip3 install numpy && apt install git
RUN git clone https://github.com/RaminHasibi/pdp_utils.git

WORKDIR /home/pdp_utils
RUN python3 setup.py install

WORKDIR /home/pdp_tw

COPY /src /home/pdp_tw/src

CMD [ "bash" ]
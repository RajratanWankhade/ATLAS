FROM ubuntu:23.04
# install pip
RUN apt-get -yq update && apt-get install -yq python3-pip python3-venv
# VENV
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install dependencies:
COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /app
# copy files into the app folder
COPY check_pytorch.py .
COPY my_network_infer.py .
COPY unet.py .
COPY my_network.pt .
# launch a bash shell
CMD bash

FROM nvcr.io/nvidia/cuda:12.6.2-base-ubuntu24.04

ENV PYTHON_VERSION 3.11
ENV MAKEFLAGS "-j$(nproc)"

RUN apt-get update &&  \
    DEBIAN_FRONTEND=noninteractive apt-get -y install htop sed wget curl rsync vim psmisc procps git mosh tmux net-tools build-essential unzip zlib1g-dev libxml2 iotop libaio-dev libbz2-dev libcap2-bin libcap-dev libtiff6 libtiff5-dev libopenjp2-7 dropbear cmake make libssl-dev libreadline-dev libsqlite3-dev ca-certificates llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev mecab-ipadic-utf8 && \
    apt-get clean && \
    rm /var/lib/apt/lists/*_*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt /opt/requirements.txt
COPY ./entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

RUN mkdir /athena
RUN chgrp -R 0 /athena

# Set-up necessary Env vars for PyEnv
ENV PYENV_ROOT /athena/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install pyenv
RUN set -ex \
    && curl https://pyenv.run | bash \
    && pyenv update \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pyenv rehash

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /opt/requirements.txt

RUN chgrp 0 /etc/passwd \
  && chmod g+w /etc/passwd
RUN chgrp 0 /etc/group \
  && chmod g+w /etc/group

# You can modify the CMD statement as needed....
ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["/bin/bash", "-l"]

ARG IMAGE
FROM ${IMAGE}

COPY update_sources.sh /
RUN /update_sources.sh

RUN dpkg --add-architecture arm64
RUN apt-get update && apt-get install -y \
  sudo \
  debhelper \
  build-essential \
  crossbuild-essential-arm64 \
  libusb-1.0-0-dev \
  libusb-1.0-0-dev:arm64 \
  libglib2.0-dev \
  libglib2.0-dev:arm64 \
  libgstreamer1.0-dev \
  libgstreamer1.0-dev:arm64 \
  libgstreamer-plugins-base1.0-dev \
  libgstreamer-plugins-base1.0-dev:arm64 \
  libgtk-3-dev \
  libgtk-3-dev:arm64 \
  python \
  python3-all \
  python3-six \
  zlib1g-dev \
  zlib1g-dev:arm64 \
  pkg-config \
  zip \
  unzip \
  curl \
  wget \
  git \
  subversion \
  vim \
  python3-numpy

ARG BAZEL_VERSION=4.0.0
RUN wget -O /bazel https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    bash /bazel && \
    rm -f /bazel

RUN svn export https://github.com/google-coral/edgetpu/trunk/libedgetpu /libedgetpu


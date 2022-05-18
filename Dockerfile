FROM arm32v7/python:bullseye

ENV DEBIAN_FRONTEND noninteractive

RUN set -e \
      && ln -sf bash /bin/sh

RUN set -e \
      && pip install -U --no-cache-dir \
        https://github.com/dceoy/macdf/archive/main.tar.gz

ENTRYPOINT ["/usr/local/bin/macdf"]

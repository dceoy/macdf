FROM dceoy/tifft:latest

ADD https://github.com/dceoy/oanda-cli/archive/master.tar.gz /tmp/oanda-cli.tar.gz
ADD https://github.com/dceoy/tifft/archive/main.tar.gz /tmp/tifft.tar.gz
ADD . /tmp/macdf

RUN set -e \
      && pip install -U --no-cache-dir /tmp/*.tar.gz /tmp/macdf \
      && rm -rf /tmp/*.tar.gz /tmp/macdf

ENTRYPOINT ["/usr/local/bin/macdf"]

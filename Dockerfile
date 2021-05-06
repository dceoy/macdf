FROM dceoy/oanda-cli:latest

ADD https://github.com/dceoy/tifft/archive/masin.tar.gz /tmp/tifft.tar.gz
ADD https://github.com/dceoy/oanda-cli/archive/master.tar.gz /tmp/oanda-cli.tar.gz
ADD . /tmp/macdf

RUN set -e \
      && pip install -U --no-cache-dir /tmp/oanda-cli.tar.gz /tmp/tifft.tar.gz /tmp/macdf \
      && rm -rf /tmp/*.tar.gz /tmp/macdf

ENTRYPOINT ["/usr/local/bin/macdf"]

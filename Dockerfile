FROM nginxinc/nginx-unprivileged:1-alpine

RUN apk update
RUN apk upgrade
RUN apk add python3
RUN apk add py-pip

RUN pip install mkdocs
RUN pip install mkdocs-material

WORKDIR /develop_floriane
COPY . /develop_floriane

WORKDIR /.
RUN mkdir /mkdocs/
RUN mkdir /mkdocs/docs/
RUN mkdir /mkdocs/docs/azure/
RUN mkdir /mkdocs/docs/webviz/
RUN mkdir /mkdocs/site/

WORKDIR /mkdocs/docs/

COPY about.md .
COPY contact.md .
COPY tutorials.md .
COPY updates.md .

WORKDIR /mkdocs/docs/azure/
COPY ert.md .
COPY get-started.md .

WORKDIR /mkdocs/docs/webviz/
COPY overview.md .

WORKDIR /mkdocs/docs/webviz/maps/
COPY mig-time.md .
COPY agg-map.md .
COPY mass-map.md .
COPY theory.md .


WORKDIR /mkdocs/
COPY mkdocs.yml .
RUN mkdocs build
RUN rm -Rf /mkdocs/docs

COPY nginx.conf /etc/nginx/conf.d/default.conf

RUN chown -R 1001:1001 /mkdocs
USER 1001

EXPOSE 8000
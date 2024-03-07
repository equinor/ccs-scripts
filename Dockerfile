FROM python:3.11-alpine3.18
FROM nginxinc/nginx-unprivileged:1-alpine

# Environment variables
ENV PACKAGES=/usr/local/lib/python3.11/site-packages
ENV PYTHONDONTWRITEBYTECODE=1

RUN python -m pip install mkdocs
RUN python -m pip install mkdocs-material

WORKDIR /develop_floriane
COPY . /develop_floriane

# Copy files necessary for build
#COPY material material
#COPY package.json package.json
#COPY README.md README.md
#COPY *requirements.txt ./
#COPY pyproject.toml pyproject.toml


# Set working directory
COPY /docs/about.md /docs/about.md
COPY /docs/contact.md /docs/contact.md
COPY /docs/tutorials.md /docs/tutorials.md
COPY /docs/updates.md /docs/updates.md

#Azure
COPY /docs/azure/ert.md /docs/azure/ert.md
COPY /docs/azure/get-started.md /docs/azure/get-started.md
COPY /docs/azure/ert-config.jpg /docs/azure/ert-config.jpg

#Stylesheet
COPY /docs/stylesheets/extra.css /docs/stylesheets/extra.css
COPY /docs/stylesheets/neoteroi-cards.css /docs/stylesheets/neoteroi-cards.css
COPY /docs/stylesheets/Equinor_Symbol_Favicon_RED_32x32px.png /docs/stylesheets/Equinor_Symbol_Favicon_RED_32x32px.png
COPY /docs/stylesheets/Equinor_Symbol_Favicon_RED_64x64px.png /docs/stylesheets/Equinor_Symbol_Favicon_RED_64x64px.png


#Webviz
COPY /docs/webviz/overview.md /docs/webviz/overview.md

#Webviz maps
COPY /docs/webviz/maps/mig-time.md /docs/webviz/maps/mig-time.md
COPY /docs/webviz/maps/agg-map.md /docs/webviz/maps/agg-map.md
COPY /docs/webviz/maps/mass-map.md /docs/webviz/maps/mass-map.md
COPY /docs/webviz/maps/theory.md /docs/webviz/maps/theory.md
COPY /docs/webviz/maps/agg-map.jpg /docs/webviz/maps/agg-map.jpg
COPY /docs/webviz/maps/grid_aggregation.jpg /docs/webviz/maps/grid_aggregation.jpg
COPY /docs/webviz/maps/mass-map.jpg /docs/webviz/maps/mass-map.jpg
COPY /docs/webviz/maps/gridding_approach.png /docs/webviz/maps/gridding_approach.png
COPY /docs/webviz/maps/mig-time.jpg /docs/webviz/maps/mig-time.jpg




#Webviz plugins
COPY /docs/webviz/plugin/co2-leakage.md /docs/webviz/plugin/co2-leakage.md

#Webviz scripts
COPY /docs/webviz/scripts/plume-extent.md /docs/webviz/scripts/plume-extent.md
COPY  /docs/webviz/scripts/plume-area.md /docs/webviz/scripts/plume-area.md
COPY /docs/webviz/scripts/co2-containment.md /docs/webviz/scripts/co2-containment.md


WORKDIR /.
COPY mkdocs.yml .
RUN mkdocs build
RUN rm -Rf /docs

COPY nginx.conf /etc/nginx/conf.d/default.conf

#RUN chown -R 1001:1001 /develop_floriane
#USER 1001

RUN RUN useradd -u 8877 develop_floriane
USER 8877

# Expose MkDocs development server port
EXPOSE 8000

# Start development server by default
#ENTRYPOINT ["mkdocs"]
#CMD ["serve", "--dev-addr=0.0.0.0:8000"]
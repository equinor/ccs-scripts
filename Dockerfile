# Copyright (c) 2016-2024 Martin Donath <martin.donath@squidfunk.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

FROM python:3.11-alpine3.18

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

#Stylesheet
COPY /docs/stylesheets/extra.css /docs/stylesheets/extra.css
COPY /docs/stylesheets/neoteroi-cards.css /docs/stylesheets/neoteroi-cards.css

#Webviz
COPY /docs/webviz/overview.md /docs/webviz/overview.md

#Webviz maps
COPY /docs/webviz/maps/mig-time.md /docs/webviz/maps/mig-time.md
COPY /docs/webviz/maps/agg-map.md /docs/webviz/maps/agg-map.md
COPY /docs/webviz/maps/mass-map.md /docs/webviz/maps/mass-map.md
COPY /docs/webviz/maps/theory.md /docs/webviz/maps/theory.md


WORKDIR /.
COPY mkdocs.yml .
RUN mkdocs build
RUN rm -Rf /docs

COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose MkDocs development server port
EXPOSE 8000

# Start development server by default
ENTRYPOINT ["mkdocs"]
CMD ["serve", "--dev-addr=0.0.0.0:8000"]
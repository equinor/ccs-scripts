FROM mcr.microsoft.com/devcontainers/python:3-3.12-bookworm

ENV PATH="/home/vscode/.local/bin:${PATH}"

# Install libegl1 library, which is required py PyQt, which is
# required by xtgeo (at least for now). The final rm -rf command
# is just standard housekeeping to reduce image size.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libegl1 \
    && rm -rf /var/lib/apt/lists

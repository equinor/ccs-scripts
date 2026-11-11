FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

ENV PATH="/home/vscode/.local/bin:${PATH}"

# The unused Yarn repository in the base image has a stale signing key.
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends libegl1 \
    && rm -rf /var/lib/apt/lists

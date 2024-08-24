 # Use the Python 3.11 slim base image
FROM python:3.11-slim 

ARG DEBIAN_FRONTEND="noninteractive"

# Define a non-root user for the container
ARG NON_ROOT_USER="nonroot"  

# Set the user ID for the non-root user
ARG NON_ROOT_UID="2222"  

# Set the group ID for the non-root user
ARG NON_ROOT_GID="2222"  

# Set the home directory for the non-root user
ARG HOME_DIR="/home/${NON_ROOT_USER}"  

# Set the directory to copy from
ARG REPO_DIR="."  

# Create the non-root user with the specified user ID
RUN useradd -l -m -s /bin/bash -u ${NON_ROOT_UID} ${NON_ROOT_USER} 

# Update package lists, install curl and git, and clean up the package cache 
RUN apt update && \
    apt -y install curl git && \
    apt clean 

ENV PIP_PREFER_BINARY=1

ENV PYTHONIOENCODING utf8  
ENV LANG "C.UTF-8"  
ENV LC_ALL "C.UTF-8"  

# Add the local bin directory to the PATH
ENV PATH "./.local/bin:${PATH}"  

# Switch to the non-root user
USER ${NON_ROOT_USER}  

# Set the working directory to the home directory of the non-root user
WORKDIR ${HOME_DIR}  

# Copy the contents of the specified directory to the container's working directory, preserving ownership and permissions
# Copy only the requirements.txt initially to leverage Docker cache
# COPY <source> <destination>

COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} ${REPO_DIR}/requirements.txt outfit_oracle/requirements.txt

# Install pip requirements
RUN pip install -r outfit_oracle/requirements.txt

# After pip install, copy the rest of the repository
COPY --chown=${NON_ROOT_USER}:${NON_ROOT_GID} ${REPO_DIR} outfit_oracle

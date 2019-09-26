TAG=kuaa/kuaa

WORKDIR=$(shell pwd)
DOCKER_XCONFIG=--env=DISPLAY=$$DISPLAY --volume='/tmp/.X11-unix:/tmp/.X11-unix:rw'
DOCKER_DATA_VOLUME?=--volume=$(WORKDIR):/data

build_container:
	docker build --rm=true -t $(TAG) .


# Run the container
# KUAA_DATA_VOLUME should contain the docker options to add an addition volume to store data
run_container:
	xhost + # Unsafe way to allow X server
	cd / && \
	docker run --rm=true --privileged -it $(DOCKER_XCONFIG) $(DOCKER_DATA_VOLUME) $(TAG) # A change of directory is mandatory to allow volumes anywhere



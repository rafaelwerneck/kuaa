TAG=kuaa/kuaa

build_container:
	docker build --rm=true -t $(TAG) .

run_container:
	docker run --rm=true --privileged -it $(TAG)

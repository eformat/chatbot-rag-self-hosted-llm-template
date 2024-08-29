# Image URL to use all building/pushing image targets
REGISTRY ?= quay.io
REPOSITORY ?= $(REGISTRY)/eformat/chatbot-rag

IMG := $(REPOSITORY):latest
VERSION := v0.0.1

# podman Login
podman-login:
	@podman login -u $(PODMAN_USER) -p $(PODMAN_PASSWORD) $(REGISTRY)

# Push for Release
podman-push-release:  podman-tag-release
	@podman push $(REPOSITORY):$(VERSION)
	@podman push $(REPOSITORY):latest

# Build the podman image
podman-build:
	@podman build --platform linux/amd64 . -t ${IMG} -t ${IMG}-x86_64 -f Dockerfile
#	@podman build --platform linux/arm64 . -t ${IMG}-aarch64 -f Dockerfile

# Push the podman image
podman-push: podman-build
	@podman push ${IMG}
	@podman push ${IMG}-x86_64
#	@podman push ${IMG}-aarch64

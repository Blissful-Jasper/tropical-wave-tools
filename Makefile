SHELL := /bin/bash

.PHONY: setup test docs docs-serve pages app build publish publish-test

setup:
	bash scripts/twave.sh setup

test:
	bash scripts/twave.sh test

docs:
	bash scripts/twave.sh docs

docs-serve:
	bash scripts/twave.sh docs-serve

pages:
	bash scripts/twave.sh pages

app:
	bash scripts/twave.sh app

build:
	bash scripts/twave.sh build

publish:
	bash scripts/twave.sh publish

publish-test:
	bash scripts/twave.sh publish-test

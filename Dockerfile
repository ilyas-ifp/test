FROM python:3.11-alpine as compiler

RUN apk add --update --no-cache build-base gfortran cmake openblas-dev
RUN python3 -m venv ./pyenv
RUN pyenv/bin/pip install pip --upgrade
COPY requirements.txt /usr/src/app/
RUN --mount=type=cache,target=/root/.cache pyenv/bin/pip install  -r /usr/src/app/requirements.txt --extra-index-url https://alpine-wheels.github.io/index


FROM python:3.11-alpine
RUN apk --update upgrade && apk add --update --no-cache libstdc++ openblas
COPY --from=compiler /pyenv /pyenv


WORKDIR /usr/src/app

COPY . /usr/src/app

CMD ["/pyenv/bin/python", "test.py"]

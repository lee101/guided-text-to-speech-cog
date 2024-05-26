
# Text Guided Text to Speech Generation

I converted the model to half precision and updated it as a cog so it can be commercialized on replicate more easily.


## How to use

sudo cog predict -i prompt="hi hows it going" -i voice="A Well spoken english male clear voice no background noise"

![audio](output.wav)


## How to train

See original parler paper.

## Clone the model

mkdir models
cd models
git clone git@hf.co:spaces/parler-tts/parler_tts_mini

## How to convert to half precision

Uncomment code in predict.py to do that, run it and then copy missing files over from the old full precision model folder.

## How to convert to cog

I did that/thats what this repo is.

see [predict.py](predict.py)


## How to deploy

cog push

## How to run tests

```
pytest .
```

## How to run lint

```
flake8 .
```

# Plugs and sponsors For AI products.
See text to speech models on Text-generator.io [https://text-generator.io/](https://text-generator.io/)

AI Chat characters https://netwrck.com [https://netwrck.com](https://netwrck.com)

AI Art Generation https://aiart-generator.io [https://aiart-generator.io](https://aiart-generator.io)

---
title: Parler-TTS Mini
emoji: 🥖
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: 4.26.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: High-fidelity Text-To-Speech
---

I converted the model to half precision and updated it as a cog so it can be commercialized on replicate more easily.


## How to use


sudo cog run -i prompt="hi hows it going" -i voice="A Well spoken english male clear voice no background noise"


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
# Tycho

Tycho is an attempt at applying boosting techniques to a variety of models (currently only supports some custom-made "self-regularizing" weaknets) and build data-driven (i.e. generate based on the dataset) boosters that combine the stability of classical tree based boosting with the ~overfitting~ more powerful equations allowed by neural networks.

Tycho is also the python prototype for [a Rust library](https://github.com/George3d6/tycho-rs) that I aim to release at some point (I swear) in order to try and improve on the performances allowed by torch and tensorflow as well as have added compile-time safety.

Tycho is a side project that I aim to integrated into [mindsdb's lightwood](https://github.com/mindsdb/lightwood) if the results are good enough, so I'd encourage you to use that for now if you want a generic nn+torch based solution to a variety of ML problems (it also has built in support for text, images, array, dates, audio... etc, tycho only supports categorical and numerical)

## Usage

1. Add the repo to your `PYTHONPATH`
2. [See this example](test/func/simple.py)

## Configuration
For logging level set the env variable `TYCHO_LOG_LEVEL` to whatever log level you want. | Default: DEBUG

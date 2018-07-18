**This repo is a work-in-progress status without code cleanup and refactoring.**

## Introduction
This is an implementation of a paper [Polyphonic Music Generation with Sequence Generative Adversarial Networks](https://arxiv.org/abs/1710.11418) in TensorFlow.

Hard-forked from the [official SeqGAN code](https://github.com/LantaoYu/SeqGAN).

## Requirements
Python 2.7

Tensorflow 1.4 or newer (tested on 1.9)

pip packages: music21, pyyaml, nltk, pathos

## How to use
`python music_seqgan.py` for full training run.

SeqGAN.yaml contains (almost) all hyperparameters that you can play with.

5 sample MIDI sequences are automatically generated per epoch.

## Dataset
The model uses a MIDI version of Nottingham database (<http://abc.sourceforge.net/NMD/>) as the dataset.

Preprocessed musical word tokens are included in the "dataset" folder.


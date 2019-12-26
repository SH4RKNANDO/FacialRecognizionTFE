#!/bin/bash

PACKAGES="python-virtualenvwrapper python-virtualenv python-numpy opencv opencv-samples cmake"

pacman -Syy

for p in $PACKAGES
do
  echo -e "\nInstall package : $p\n"
  sudo pacman -S $p

done


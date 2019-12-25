#!/bin/bash

function check_system {

  OS=$(awk '/DISTRIB_ID=/' /etc/*-release | sed 's/DISTRIB_ID=//' | tr '[:upper:]' '[:lower:]')
  ARCH=$(uname -m | sed 's/x86_//;s/i[3-6]86/32/')
  VERSION=$(awk '/DISTRIB_RELEASE=/' /etc/*-release | sed 's/DISTRIB_RELEASE=//' | sed 's/[.]0/./')

  if [ -z "$OS" ]; then
      OS=$(awk '{print $1}' /etc/*-release | tr '[:upper:]' '[:lower:]')
  fi

  if [ -z "$VERSION" ]; then
      VERSION=$(awk '{print $3}' /etc/*-release)
  fi

  echo $OS
  echo $ARCH
  echo $VERSION
}

check_system
exit

PACKAGES="python-virtualenvwrapper python-virtualenv python-numpy opencv opencv-samples"

pacman -Syy

for p in $PACKAGES
do
  echo -e "\nInstall package : $p\n"
  sudo pacman -S $p
done

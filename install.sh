#!/bin/bash

env="source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate"
name=$(hostname)

if ! grep -q "$env" ~/.bashrc; then
    echo $env | sudo tee -a ~/.bashrc > /dev/null
fi

source ~/.bashrc

if [ ! -f influxdata-archive_compat.key ]; then
    wget -q https://repos.influxdata.com/influxdata-archive_compat.key
    echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
    echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list

    sudo apt-get update && sudo apt-get install influxdb2 influxdb2-cli -y
    sudo systemctl start influxdb

    influx setup \
      --username $name \
      --org "ucberkeley" \
      --bucket "lab6" \
      --force
fi
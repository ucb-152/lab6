# Neuron Profile User Guide

## Installing InfluxDB on DLAMI Trn1
To install InfluxDB compatible with `neuron-profile`:
```bash
wget -q https://repos.influxdata.com/influxdata-archive_compat.key
echo '393e8779c89ac8d958f81f942f9ad7fb82a25e133faddaf92e15b16e6ac9ce4c influxdata-archive_compat.key' | sha256sum -c && cat influxdata-archive_compat.key | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg > /dev/null
echo 'deb [signed-by=/etc/apt/trusted.gpg.d/influxdata-archive_compat.gpg] https://repos.influxdata.com/debian stable main' | sudo tee /etc/apt/sources.list.d/influxdata.list

sudo apt-get update && sudo apt-get install influxdb2 influxdb2-cli -y
sudo systemctl start influxdb
influx setup
```

For the setup parameters, enter
- Username: ubuntu
- Organization: ucberkeley
- Bucket: primary_bucket

For all other parameters, just click enter to use the default values.

## Using neuron-profile
To create the NEFF and NTFF files to run `neuron-profile` with, add the decorator below to the target kernel:
```python
@nki.profile(working_directory="/home/ubuntu/", save_neff_name='nki_kernel.neff', save_trace_name='nki_kernel.ntff', profile_nth=2)
```
Replace the parameters with the names and paths of your choice.

Then, run the kernel to generate the files. Note that the kernel will no longer return values, so disable any verification check you have when you have the profile decorator activated. Replace the python file with the file that calls your nki.profile decorated kernel.
```bash
python program.py
```

Finally, use `neuron-profile` to view the profile GUI.
```bash
neuron-profile view -n nki_kernel.neff -s nki_kernel_exec_2.ntff --db-bucket=my_kernel
```
This will take a while to load, but once it does, click on the link to view the profile GUI.

Make sure you have port forwarding enabled. You can run this command on a seperate terminal on your local machine to enable port forwarding.
```bash
ssh trn1_cs152 -L 3001:localhost:3001 -L 8086:localhost:8086
```





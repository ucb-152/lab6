# Neuron Profile User Guide
Neuron Profile is an AWS tool to allow users to view the performance metrics of programs run on Neuron Devices like Tranium. Read the [Neuron Profile User Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html) for more detailed information.

## Installing InfluxDB on DLAMI Trn1

> [!NOTE]
>
> If you ran the `install.sh` script, this step as already been completed, so you can move on to [Using neuron-profile](#using-neuron-profile) section.

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
- Bucket: lab6

For all other parameters, just click enter to use the default values.

## Using neuron-profile

### Generating NEFF and NTFF Files
In order to view the execution profile of a NKI kernel you need the NEFF and NTFF files. The NEFF file is the compiled instructions from the python kernel that are actually executed on the NeuronCore. The NTFF file is used to record various metrics of the memory and compute engines while profiling the execution of the kernel. Here are various ways to generate NEFF and NTFF files from NKI kernels.

> [!NOTE]
>
> If you generated the profile files using `tester.py --profile`, you can skip to the [Viewing Profiles](#viewing-profiles) section. The `tester.py` script will have saved the NTFF/NEFF files in the `nki_conv2d/profiles` folder.

#### NKI Execution Functions
To generate a `.neff` file, you can modify how you call the NKI kernel in python as shown below
```python
bench_func = nki.benchmark(
    warmup=5, iters=20, save_neff_name=f"file_name.neff"
)(kernel)
bench_func(*args)
```
Here is the full description of how to use [nki.benchmark](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.benchmark.html#nki.benchmark). You can generate the NEFF file similarly with [nki.profile](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.profile.html#nki.profile) and [nki.baremetal](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.baremetal.html#nki.baremetal).

#### NKI Decorators
You can also add these parameters directly to the kernel by changing the python decorator, but usually its easier to leave the decorator as @nki.jit, and choose between normal execution, benchmarking, and profiling in your top-level module or tester script that calls the kernel. Nevertheless, here is how to do it with nki.profile.

Add the decorator below to the target kernel, replacing the `@nki.jit` decorator.
```python
@nki.profile(working_directory="/home/ubuntu/", save_neff_name='nki_kernel.neff', save_trace_name='nki_kernel.ntff', profile_nth=2)
```
Replace the parameters with the names and paths of your choice. Note that if you specify `profile_nth`, the trace file will be saved to `{save_trace_name - .ntff}_exec_{profile_nth}.ntff`. So in this case, the trace will be saved to `nki_kernel_exec_2.ntff`.

Then, run the kernel with your top-level module ot tester script to generate the files. Note that the kernel will no longer have any return values, so disable any verification check you have when you have the profile decorator activated. 
```bash
python program.py
```

#### Generating NTFF File from NEFF file
The NEFF file will be generated, and the NTFF file if you specify `save_trace_name`. If you don't specify `save_trace_name`, you can generate the NTFF file by profiling the NEFF file execution with the command below.
```bash
neuron-profile capture -n <neff_file_name> -s <ntff_file_name>
```

### Viewing Profiles
Now, use `neuron-profile` to view the profile GUI.
```bash
neuron-profile view -n <file name>.neff -s <file name>.ntff
```
This will take a while to load, but once it does, it will output a localhost link you can click to view the GUI.

Make sure you have port forwarding enabled. You can run this command in a seperate terminal on your **local machine** to enable port forwarding:
```bash
ssh trn1_cs152 -L 3001:localhost:3001 -L 8086:localhost:8086
```

### Interpretting the Neuron Profile GUI
It is suggested to read the [Neuron Profile User Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html) to better understand how to interact with the Neuron Profile GUI and interpret the results.

Here are some steps you can take to interpret the profile:
- We suggest hiding the DMA profiling rows initially to first understand the utilization of the compute engines. Select 'View Settings', and toggle 'Show dma' off, then click the orange 'Save' button. You can re-enable the dma rows later if you want to look into the data movement.
- Zoom in on a short time period by selecting a narrow horizontal range on the timeline. Click at the top of the profile and drag to the right to select the narrow time range. This will allow you to better visualize the utilization of the compute engines
- You can also view the summary statistics by clicking on the "Summary." Look at "overall_stats > hardware_flops", "tensor_engine > tensor_engine_active_time_percent", and other statistics to get a better understand of how your kernel is using the hardware.

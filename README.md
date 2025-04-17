# CS152 Laboratory Exercise 6

## Introduction and Goals:
The goal of this laboratory assignment is to give you an opportunity to program hardware accelerators. As the computational demands of AI and ML applications continue to increase, industry and research efforts have been attempting to meet these demands with Domain Specific Acceleration and custom accelerator hardware. As a result, an increasingly important skill is the ability to map software applications and kernels onto new architectures. 

It is important to learn how to optimize programs to take full advantage of the memory and compute engines available on the target hardware. There are many factors to consider when designing a kernel, such as the communication between compute engines and memory, the amount of data a compute engine can do work on at a given time, the dependencies between different computations in your kernel, and more. By the end of this lab, you should be able to program basic kernels on [NeuronCore](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuron-core-v2.html#neuroncores-v2-arch), the main accelerator device in [AWS Tranium](https://aws.amazon.com/ai/machine-learning/trainium/) machines.

### Graded Items:
All reports are to be submitted through [Gradescope](https://www.gradescope.com/courses/959486)

> [!IMPORTANT] 
> 
> TODO: Add details on assignment here:
> - Link to Gradescope Assignment
> - Directed vs Open-Ended split?
> - Submission details for each question
> - Page limits


## Background:
### ML Accelerator Hardware
In previous labs and lectures, you have learned about various architectures like CPUs (scalar, superscalar, out-of-order, VLIW, etc.), Vector Engines, GPUs, etc. One type of architecture that is prevalent in ML accelerator chips are **systolic arrays**. Systolic array architectures are often used in ML and AI applications because they are suited for computing operations on matrices, which are often used in applications like Neural Networks and Large Language Models. A systolic array is a type of spatial array, which describes a family of architectures that use arrays of compute cells called PEs (Processing Elements), which typically do MAC (Multiply and Accumulate) operations.

<p align="center">
  <img width="400" src="./img/systolic-array-arch.webp">
  <br>
  <a href="https://www.mdpi.com/2079-9292/13/8/1500">Source</a>
</p>

Systolic arrays are good for kernels like matrix multiplication, as they have high data reuse. From the animation below, you can see that one 3 by 3 matrix is loaded into the PE array, while the other 3 by 9 matrix streams into the PEs. The output 3 by 9 matrix is streamed out of the systolic array after 3 MAC operations (since for a 3x3 by 3x9 matrix multiplication, each row by column inner product is 3 multiply and add operations).

<p align="center">
  <img width="400" src="./img/systolic-array-matmul-animation.gif">
  <br>
  <a href="https://medium.com/lightmatter/matrix-processing-with-nanophotonics-998e294dabc1">Source</a>
</p>


### Tranium Overview
In this lab, we will work on AWS Tranium. These instances contain a single Tranium Device, which has 2 NeuronCores. Each NeuronCore has a HBM (High-bandwidth memory) unit and on-chip storage units that the compute units interface with. Each core has various compute units optimized for different functions:
- Tensor Engine: 128 x 128 systolic array for matrix operations
- Vector Engine: 128-wide vector unit, reductions, dependent calculations
- Scalar Engine: 128-wide scalar unit, for activation functions, independent calculations
- GpSimd Engine: general-purpose engine for operations not suited for the other engines

The NeuronCores are highly optimized and designed for ML workloads, and thus each engine has its own unique purpose in common ML algorithms. We can get maximal performance out of the chip if we carefully integrate the engines together, making sure that our algorithm is designed such that computations are mapped to the appropriate engines while accounting for bandwidth and throughput constraints for the engines, their internal connections, and the memory system.

<p align="center">
  <img width="400" src="./img/neuron_device2.png">
  <br>
  <a href="https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/arch/trainium_inferentia2_arch.html#trainium-inferentia2-arch">Source</a>
</p>

There are various levels of memory at play in the Tranium Instance. There is the host memory that is external to the Neuron Cores. Then, there is the HBM, which is the main on-device memory. Finally there is the on-chip memory, consisting of the SBUF (State Buffer) and the PSUM (Partial Sum Buffer). The levels, sizes, and bandwidths of these memories are shown below.

<p align="center">
  <img width="600" src="./img/memory_hierarchy.png">
  <br>
  <a href="https://github.com/stanford-cs149/asst4-trainium/tree/main">Source</a>
</p>

All computations require loading data from the HBM into the SBUF, which is connected as an input to all of the engines. The output of the Tensor Engine is stored in the PSUM, which can be an input to the Vector and Scalar Engines. The Vector, Scalar, and GpSimd engine can write back to the SBUF. 

<p align="center">
  <img width="400" src="./img/neuron_core.png">
  <br>
  <a href="https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/arch/trainium_inferentia2_arch.html#trainium-inferentia2-arch">Source</a>
</p>

There are a lot of factors at play when writing kernels on Tranium devices, and good kernels will take advantage of the all of the compute engines and full memory heirarchy to reduce bottlenecks and extract the most performance. For more details on Tranium architecture, look at the [Tranium Architecture Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/arch/trainium_inferentia2_arch.html#trainium-inferentia2-arch).


### Tranium Setup
To begin working on Tranium, follow the instructions in [AWS_SETUP.md](/AWS_SETUP.md)

> [!IMPORTANT] 
> 
> Do not proceed with the rest of the lab without completing this step.


## Directed Portion
For the Directed portion, you are tasked with developing a `ffnn` (Feedforward Neural Network) kernel on Tranium. The goal of this task is to familarize yourself with the Tranium and NeuronCore architecture, and how to program them using AWS's Neuron Kernel Interface or NKI, which you will learn more about soon.

### Overview of Feedforward Neural Networks

<p align="center">
  <img width="400" src="./img/feedforward_nn.jpg">
  <br>
  <a href="https://www.geeksforgeeks.org/feedforward-neural-network/">Source</a>
</p>

A feedforward neural network is a type of neural network where the information flows through the layers in one direction. We start with the input layer, which recieves the initial data, with each neuron acting as a feature of the input data. 

Then, we pass through multiple hidden layers, and each neuron applies a weighted sum of its inputs, often with an added bias, followed by an activation function. This is calculation is often expressed as a matrix multiplication. The input matrix `X` has dimensions `(N, d)`, where `N` is the number of samples (often referred to as the batch size), and `d` is the number of input features. Each connection between layers can be represented by a weight matrix `W` and a bias vector `b`. The `W` matrix has dimensions `(d, h)`, where `h` is the number of neurons in the hidden layer. The bias vector `b` has a length of `h`. 

You can calculate the activations of the next layer of neurons with the equation `H = ACT(XW+b)`, where `ACT` is some activation function like [ReLu](https://www.geeksforgeeks.org/relu-activation-function-in-deep-learning/)(typically used on the hidden layers) or [Softmax](https://www.geeksforgeeks.org/the-role-of-softmax-in-neural-networks-detailed-explanation-and-applications/) (typically used on the output layer). The bias vector is broadcasted to the dimensions `(d, h)` and thus the bias vector is added to each row of the `XW` matmul result. We can continue with similar caculations for each layer, until we reach the output layer. This process of taking the inputs and stepping through the layers of the neural network, until we reach the output layer, is known as the **forward pass**.

The output layer contains the activations of the neurons that correspond to the output of the neural network. For example, in a classification problem, each neuron could correspond to a class, and the neuron with the highest activation (i.e. the highest probability) would be the class that the neural network is **predicting** for the input.  

If you would like to learn more on feedforward neural networks, check out this article on [Feedforward Neural Networks](https://www.geeksforgeeks.org/feedforward-neural-network/). 

### Step 0: Setup
To begin, ssh into your Tranium instance or open a remote session using VSCode (or another application). 

Once you have logged into the instance, clone the lab repo. 
```bash
git clone <TODO: Insert repo link here>
cd lab6
```

Finally, run the `install.sh` script. 
```bash
source install.sh
```
The install script will activate the Python virtual environment prebuilt on the AWS instances with the Deep Learning AMI (`source /opt/aws_neuronx_venv_pytorch_2_5/bin/activate`), that contains all the dependencies needed for the assignment. It will also modify your ~/.bashrc file so that the virtual environment is activated automatically upon future logins to your machine. Finally, the script sets up your InfluxDB credentials so that you may use neuron-profile, which will be useful for future sections.

All files needed for the directed portion are located in `lab6/nki_ffnn`.
- `utils.py`: Utility functions for loading the matrices, and constants for the matrix dimensions
- `ffnn_ref.py`: Reference NumPy implementation of the Feedforward Neural Network.
- `ffnn.py`: Main program to run the kernels and benchmark performance.
- `matmul_kernels.py`: Matrix Multiplication kernels developed by AWS, with various levels of optimization. 
  - Read the [AWS Matrix Multiplication tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/tutorials/matrix_multiplication.html#matrix-multiplication) for more information. 
- `kernels.py`: Contains the kernels you will need to implement for the FFNN. 
  - **This is the only file you will need to edit.**
- `tester.py`: Debug kernels individually on CPU before running full kernel on Tranium.

### Step 1: Observe the Python/Numpy Reference FFNN

To start, first take a look at `ffnn_ref.py` for a Numpy implementation of the Feedforward Neural Network. This will give you a valuable insight into the operations and kernels needed to perform this computation. Then, run the following command to benchmark the reference implementation.
```bash
python ffnn_ref.py --benchmark
```
You should see that the prediction operation takes roughly 440-450ms to run using Python and Numpy. Keep this figure in mind when comparing to the performance of the kernel on Tranium using NKI.

Now, run the program again with the following command-line flags to store the input data and golden model results. 
```bash
python ffnn_ref.py --store-data
```
There should be `*.bin` files in the `ffnn` directory, one for each of the following matrices: `X`, `W1`, `b1`, `W2`, `b2`, and `Y`. We will use these for running and verifying the NKI implementation.

### Step 2: Learn about Neuron Kernel Interface(NKI)
In order to program the Tranium devices easily, we will take advantage of AWS's [Neuron Kernel Interface](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html) or NKI. This is a collection of APIs that allow users to program directly in Python and perform computations using the Tranium engines.
> [!IMPORTANT]
>
> Make sure to skim through the [Neuron Kernel Interface](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html) documentation, and pay particular attention to the [NKI Language](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.language.html) APIs.
> 
> Also, make sure to read these sections of the documentation before proceeding:
> - [Implementing your first NKI kernel](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/getting_started.html#implementing-your-first-nki-kernel)
> - [Representing data in NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/programming_model.html#representing-data-in-nki)
>
> In addition to the linked sections, we highly recommend reading or skimming the full guides, as they will help in developing the NKI kernels.

An important detail is that NKI operations often have dimension restrictions due to the physical limits of the hardware. Thus, we must "tile" our operations when dealing with larger matrices. Tiling is quite common in ML workloads and kernels, as the inputs are very, very large. Make sure you have read the APIs carefully for the dimension restrictions, and tile your kernels accordingly. 


### Step 3: Program the nki_transpose kernel
In this part, we will develop the tranpose kernel, which will allow us to use the matmul kernels that expect a transposed input. As mentioned in the guides, there are three main stages to programming a NKI kernel: 1) Load inputs, 2) Perform computation, and 3) Store outputs. For `nki_tranpose`, we are not really performing any computation, but we can consider "transposing" as the desired modification to the input data. 

Fill in the blanks to implement the `nki_transpose` kernel in `kernels.py`. 
- Hint 1: there is a NKI API that does a combined load and transpose of a tile.
- Hint 2: Use `nl.tile_size.pmax` to get the max partition dimension. Remember, the partition dimension is the first index unless otherwise specified ([Representing data in NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/programming_model.html#representing-data-in-nki)).
- Hint 3: Use iterators to loop through the indices when tiling: [NKI Language (Iterators)](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.language.html#iterators)

Once you have completed the kernel, run the following command to confirm your implementation works:
```bash
python tester.py --test-transpose
```

If you are not passing the tests, make sure to read the documentation carefully for the limits and restrictions on various NKI APIs. 

You may also wish to print your tensor values within the NKI kernel. While this is not possible when running the kernel on Tranium, the `tester.py` script uses the `nki.simulate_kernel` feature, which enables device printing with [nl.device_print](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.language.device_print.html). Thus, you can put nl.device_print statements in your kernel if you are testing via the `tester.py` script.


### Step 4: Program the nki_bias_add_act kernel
As the name suggests, this kernel will take an input tensor, a bias vector, and an activation function, and apply the bias and activation to each row of the input tensor. 

Fill in the blanks to implement the `nki_bias_add_act` kernel in `kernels.py`. 
- Hint 1: Many of the [NKI math operations](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.language.html#math-operations) allow for the operands to have different dimensions, as long as one can be broadcasted into the other.
- Hint 2: Most common activation functions are available in the [NKI math operations](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.language.html#math-operations)

Once you have completed the kernel, run the following command to confirm your implementation works:
```bash
python tester.py --test-bias-add-act
```

### Step 5: Program the nki_forward kernel
Similar to the reference numpy model, this kernel will combine the transpose, matmul, and bias/activation kernels to perform the forward pass of the neural network. Fill in the blanks to complete the kernel. Do not change the existing skeleton code for selecting the specific matmul kernel version to use, this will be needed for benchmarking.

Fill in the blanks to implement the `nki_forward` kernel in `kernels.py`. 

Once you have completed the kernel, run the following command to confirm your implementation works:
```bash
python tester.py --test-forward
```

### Step 6: Program the nki_predict kernel
Now, we will combine all our kernels to get the probability distribution from the forward pass, and identify our output classes.
1. Fill in the blank to get the `probs` matrix, which corresponds to the probability of each of the output classes for each of the inputs
2. Select the index of the highest probability per input (i.e. per row), and place that in the `predictions` array. Do not use numpy.argmax or another raw python method for this. Use the NKI APIs.
3. Return the `predictions` array

Follow the steps above to implement the `nki_predict` kernel in `kernels.py`. 

- Hint 1: You don't need to program much for Step 1
- Hint 2: You may need to break Step 2 into two seperate steps: 1) identify the max values and 2) identify the indices of the max values. Both of the NKI APIs you need for this can be found in the [NKI ISA manual](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/nki.isa.html).

Once you have completed the kernel, run the following command to confirm your implementation works:
```bash
python tester.py --test-predict
```

### Step 7: Run nki_predict
Once you have completed all of the above steps, your NKI FFNN kernel should be complete! Run the command below to run all the kernels on Tranium:
```bash
python ffnn.py
```
If you get the message: "Predictions match the golden model." then you have succesfully completed the above steps, and can proceed to the next step. Otherwise, make sure to fix your kernels before proceeding.

### Step 8: Benchmark nki_predict
Run the following command to benchmark the `nki_predict` kernel, using the different matmul kernels. 
```bash
python ffnn.py --benchmark
```
- Compare the latency of the "tiled" matmul vs the reference numpy implementation. How much faster is the NKI implementation?
- Compare the latencies of the various matmul kernels. Record any trends or outliers you notice, and give a brief explanation for your observations.


## Open-Ended Portion
For the Open-Ended portion, you are tasked with developing a `conv2d` on Tranium, and optimizing it as much as possible! This assignment should be completed individually, and the most performant kernels will recieve prizes from AWS!

### Prizes:
The prizes for the best performing `conv2d` kernels are as follows:
- 1st Place: $200 Amazon gift card
- 2nd-4th Place: $100 Amazon gift card
- Amazon Echo Show

Now that you are excited to win some prizes, lets get into the task!

### Overview of 2D Convolutions:
TODO


### Program `conv2d_nki`
All of the files needed for this part are located in `lab6/nki_conv2d`

To start, take a look at `conv2d_ref.py` for the PyTorch and NumPy implementations for the 2D Convolution kernels:
- `conv2d_torch`: Built-in PyTorch implementation for [2D Convolution](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html). **Used as the golden model.**
- `conv2d_numpy`: A naive implementation using NumPy, performing the basic filter application and bias addition. **Intended as a naive functional reference model.**
- `conv2d_numpy_matmul`: A more optimized implementation using tranposing, reshaping, and matrix multiplication
- `conv2d_numpy_matmul_tiled`: Similar to the `conv2d_numpy_matmul` implementation but with tiling

The `conv2d_numpy_matmul` and `conv2d_numpy_matmul_tiled` are simply meant to serve as an idea of how to translate the `conv2d` operations into matrix multiplications. Feel free to reshape, tile, and operate on the data however you want, as long as you match the reference model.

To run the reference kernels and benchmark their performance, run the following command.
```bash
python ref_tester.py --benchmark
```

Note that the PyTorch version will be significantly faster than the NumPy versions, since it as been heavily optimized in the backend. The NumPy implementations are meant to provide a programatic reference for how to code the kernels on NKI. Moreover, the tiled NumPy version may be slightly slower, due to the reshaping and looping, but it will be faster (and required) on architectures like Tranium that are meant for tiling and parallelization.


#### Brainstorm using NumPy
If you want to brainstorm your implementation (reshaping, tiling, loading/storing, operations, etc), you can first create a reference implementation on NumPy by adding a function to `conv2d_ref.py` and the list of kernels in `ref_tester.py`. This way, before you move to programming using NKI, you can confirm that your approach is functionally correct (i.e. correct outputs). 

Feel free to comment out the other kernels to only benchmark the kernels you are modifying or developing.


## Acknowledgements
The original material for this lab was designed by Ronit Nagarapu in Spring 2025. The Tranium portion of the lab was developed with the assistance of AWS & Annapurna Labs and inspired by Stanford's CS149 Tranium assignments.
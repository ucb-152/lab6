# CS152 Laboratory Exercise 6

## Introduction:
The goal of this laboratory assignment is to give you an opportunity to program hardware accelerators. As the computational demands of AI and ML applications continue to increase, industry and research efforts have been attempting to meet these demands with Domain Specific Acceleration and custom accelerator hardware. As a result, an increasingly important skill is the ability to map software applications and kernels onto new architectures. 

It is important to learn how to optimize programs to take full advantage of the memory and compute engines available on the target hardware. There are many factors to consider when designing a kernel, such as the communication between compute engines and memory, the amount of data a compute engine can do work on at a given time, the dependencies between different computations in your kernel, and more. By the end of this lab, you should be able to program basic kernels on two targets: [Gemmini](https://github.com/ucb-bar/gemmini/tree/master) and [NeuronCore](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuron-core-v2.html#neuroncores-v2-arch) (the main accelerator device in [AWS Tranium](https://aws.amazon.com/ai/machine-learning/trainium/) machines).

### Graded Items:
All reports are to be submitted through [Gradescope](https://www.gradescope.com/courses/959486)

> [!IMPORTANT] 
> 
> TODO: Add details on assignment here:
> - Link to Gradescope Assignment
> - Directed vs Open-Ended split?
> - Submission details for each question
> - Page limits

## Background

Gemmini is a hardware accelerator that is part of the Chipyard ecosystem that we have been using throughout the previous labs of this course. You have previously seen Rocket Core (Berkeley's single-issue, in-order RISC-V processor) and BOOM (Berkeley's Out of Order RISC-V Core), both of which are RISC-V CPU architectures. By contrast, Gemmini is based around a systolic array architecture. 

<p align="center">
  <img width="400" src="./img/gemmini-system.png">
</p>

Systolic array architectures are often used in ML and AI applications because they are suited for computing operations on matrices, which are often used in applications like Neural Networks and Large Language Models. A systolic array is a type of spatial array, which describes a family of architectures that use arrays of compute cells called PEs (Processing Elements), which typically do MAC (Multiply and Accumulate) operations.

<p align="center">
  <img width="400" src="./img/systolic-array-matmul.webp">
  <br>
  <a href="https://www.mdpi.com/2079-9292/13/8/1500">Source</a>
</p>

Systolic arrays are good for kernels like matrix multiplication, as they have high data reuse. From the animation below, you can see that one 3 by 3 matrix is loaded into the PE array, while the other 3 by 9 matrix streams into the PEs. The output 3 by 9 matrix is streamed out of the systolic array after 3 MAC operations (since for a 3x3 by 3x9 matrix multiplication, each row by column inner product is 3 multiply and add operations).

<p align="center">
  <img width="400" src="./img/systolic-array-matmul-animation.gif">
  <br>
  <a href="https://medium.com/lightmatter/matrix-processing-with-nanophotonics-998e294dabc1">Source</a>
</p>

## Setup
To get started, clone the lab chipyard repo:
```bash
cd /scratch/${USER} 
source conda/etc/profile.d/conda.sh
git clone https://github.com/ucb-bar/chipyard-cs152-sp24.git cs152-lab6-sp25
cd cs152-lab6-sp25
git checkout cs152-lab6-sp25
./build-setup.sh riscv-tools --skip-toolchain --skip-firesim --skip-marshal --skip-circt
```

Run these commands everytime you want to start a new terminal
```bash
cd /scratch/${USER} 
source conda/etc/profile.d/conda.sh 
cd cs152-lab6-sp25
source env.sh 
LAB6ROOT=$PWD 
SIMDIR=${LAB6ROOT}/sims/verilator
GEMDIR=${LAB6ROOT}/generators/gemmini
TESTDIR=${GEMDIR}/software/gemmini-rocc-tests/bareMetalC/
BINDIR=${GEMDIR}/software/gemmini-rocc-tests/build/bareMetalC/
```

## Directed Portion

### Part 1: Setting up Gemmini
To start developing and running simulations on Gemmini, first build the Gemmini software tests.

```bash
cd ${GEMDIR}/software/gemmini-rocc-tests
./build.sh
```

Now, build the Gemmini simulator using verilator. 
```bash
cd ${SIMDIR}
make -j4 CONFIG=GemminiRocketConfig
```

Finally, try running the template test
```bash
cd ${SIMDIR}
make CONFIG=GemminiRocketConfig run-binary BINARY=${BINDIR}/template-baremetal
```

After sometime, you should see print statements with the actions the program is taking to move data and do computations using the Gemmini Accelerator. Take a look at `gemmini-rocc-tests/bareMetalC/template.c` to get an idea of how the C file of the binary we ran is going through the following steps:
1. Initializing Gemmini's TLB
2. Allocating the memory for the input and output matrices
3. Configuring and loading the identity matrix
4. Multiplying by the input matrix
5. Moving the Output matrix into main memory
6. Checking that the input = output

As you can tell, programming the accelerator requires not only knowledge of the programming language (in this case C), but also the special ways and instructions we must use to interact with the hardware accelerator, which we can consider as a "accelerator API". 

A large portion of ongoing development in research and industry is making the programming interface for accelerators easier, and some companies even develop plugins for their accelerator API in packages like PyTorch to allow developers to program directly in Python. However, lowering these high-level abstractions into the hardware API is still a work in progress, and getting the best performance will still involve direct programming in the hardware API.

> [!NOTE]
>
> Before proceeding, make sure to read the [Gemmini ISA overview](https://github.com/ucb-bar/gemmini/tree/master?tab=readme-ov-file#isa). Read up to the "Citing Gemmini" section.

Now, answer these questions in your lab report. You may want to look at the C header files `gemmini-rocc-tests/include/gemmini_params.h` and `gemmini-rocc-tests/include/gemmini.h`.
> [!IMPORTANT]
>
> **Question 1.** Take a look at the `gemmini_flush` function. What type of ROCC instruction does it call? What are the values for the fields that make up the ROCC instruction?
> 
> 
> **Question 2.** Notice that the `gemmini_config_ld` and `gemmini_config_st` functions only pass the size of the matrix in the horizontal dimension. Why do you think this is?
>
>
> **Question 3.** What is the data type of the input and output matrices? What are the dimensions and total size (in bytes) of each matrix?
>
>
> **Question 4** What function could we use to preload the array with a value (e.g. load the array with 15)?


### Part 2: Building Basic Tests on Gemmini
Now, you will modify the template code and rerun your tests. Follow the below steps to develop and build new bareMetalC tests (adopted from the gemmini-rocc-tests [README](https://github.com/ucb-bar/gemmini-rocc-tests?tab=readme-ov-file#writing-your-own-gemmini-tests)).

1. First, make a copy of the template C file and call it `double.c`
```bash
cd ${TESTDIR}
cp template.c double.c
```

2. Add `double` to the top of the list of tests in `gemmini-rocc-tests/bareMetalC/Makefile`.

3. Modify `double.c` to set the diagonal entries of the `Identity` matrix to `2`.

5. Adjust the correctness checks at the end of `double.c` to check and make sure that the `Out` matrix has elements that are double of their corresponding entry in the `In` matrix. Feel free to copy and modify the `is_equal` function in `gemmini-rocc-tests/include/gemmini_testutils.h` to accomplish this. You do not have to account for float elements since we are currently only working with integers.

6. Rebuild the tests with your modifications. The binary for double.c will now be in `${BINDIR}/double-baremetal`
```bash
cd ${TESTDIR}/..
./build.sh
```

7. Run the `double-baremetal` binary on the Gemmini simulator and ensure that your test passes.
```bash
cd ${SIMDIR}
make CONFIG=GemminiRocketConfig run-binary BINARY=${BINDIR}/double-baremetal
```

Once your test passes, answer the following questions in your lab report.
> [!IMPORTANT]
>
> **Question 5.** Attach a diff of your `double.c` file compared to the `template.c` file, as well as a diff of the `gemmini_testutils.h` file if you modified it. You can use `diff template.c double.c` and `git diff gemmini_testutils.h`, respectively, to get the diffs.





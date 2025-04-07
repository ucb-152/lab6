# AWS Setup Instructions
This document will help you launch a Tranium EC2 instance, configure SSH groups to allow you to connect to the instance, and set up auto-shutdown and budget notifications so you don't accidentally burn through your credits. These setup steps may seem tedious, but they will provide you with valuable experience in setting up cloud computing infrastructure, which you will likely run into in future career and research opportunities.

## Step 1: Launch Instance
The first step is to launch a Tranium EC2 instance.

1. First, log into [AWS Console us-east-2](https://us-east-2.console.aws.amazon.com/console/home?region=us-east-2#). You can click "Sign in using root user email" and sign in that way if needed.
2. Go the [EC2 service](https://us-east-2.console.aws.amazon.com/ec2/home?region=us-east-2#Overview:) (Elastic Compute Cloud) and click the orange "Launch Instance" button.
3. Fill in the following information for your instance. When you select the OS Type, you may get a pop up warning "Some of your current settings will be changed or removed if you proceed". You can ignore this and click the orange "Confirm changes" button.
    - Name: trn1_cs152
    - OS Type: Ubuntu
    - AMI: Deep Learning AMI Neuron (Ubuntu 22.04)
    - Instance Type: trn1.2xlarge
<p align="center">
  <img width="50%" src="./img/aws_setup/1_instance_info.png">
</p>

4. Scroll down, and click create new key pair. Select the following for your key pair details:
    - Key pair name: trn1_cs152
    - Key pair type: ED25519
    - Private key file format: .pem
<p align="center">
  <img width="40%" src="./img/aws_setup/2_key_setup.png">
</p>

Click "Create key pair", and the `trn1_cs152.pem` file will be downloaded to your local computer. Once the key is downloaded, move it to your `~/.ssh` directory, and make sure to set the permissions to read-only for the owner (400 sets r-- --- --- permissions for the file). 

```bash
mv Downloads/trn1_cs152.pem ~/.ssh
chmod 400 ~/.ssh/trn1_cs152.pem
```

5. Scroll down to the "Network settings" section. Confirm these are the settings (they should be selected by default)
    - Select "Create security group"
    - Select "Allow SSH traffic from" 
    - Select "Anywhere 0.0.0.0/0"

<p align="center">
  <img width="50%" src="./img/aws_setup/3_key_security_group.png">
</p>

6. Scroll down further to the "Configure storage" settings, and set it to 150 GiB or gp3 storage.
<p align="center">
  <img width="50%" src="./img/aws_setup/3_1_storage_settings.png">
</p>


7. Finally, on the right side of the screen where it says "Summary",
confirm the information is correct, and launch your instance!

<p align="center">
  <img width="40%" src="./img/aws_setup/4_summary_launch.png">
</p>

### Potential Errors:

#### Resource Request Validation:
You may get an error saying your "Instance launch failed" due to request for accessing resources needing validation. This usually gets resolved within 5 minutes, and once you get the email saying that your request has been validated, click the orange button saying "Retry failed tasks."

<p align="center">
  <img width="90%" src="./img/aws_setup/4_1_launch_fail_validation.png">
</p>

#### vCPU Capacity Limit
If you get an error saying your "Instance launch failed" due to requesting more vCPU capacity than your current vCPU limit of 0, make sure you fill out the [Google Form on Ed](https://edstem.org/us/courses/74390/discussion/6388697) to get permissions to launch Tranium instances. Email ronitnag04@berkeley.edu once you fill out the form, with the email subject "[CS152] SP25 Lab 6 AWS Google Form Submitted".

<p align="center">
  <img width="90%" src="./img/aws_setup/4_1_launch_fail_vcpu.png">
</p>


## Step 2: Setup Elastic IPs
This next step is to set up an Elastic IP for your instance. By default, the IPv4 associated with an instance changes each time you launch it. This is quite annoying since you will need to stop and start the instance constantly to save costs and credit usage. By allocating an Elastic IP and associating it with the instance, we avoid having to change your SSH config each time. 
1. After clicking "Launch Instance" from the previous step, you should have landed back in the EC2 dashboard. Scroll down on the left side of the screen until you get to "Network & Security" settings, and click on Elastic IPs.
2. Click the orange "Allocate Elastic IP address" button in the top-right corner of the screen
3. Click the orange "Allocate" button. Now, you have an Elastic IP to use.
4. Rename the Elastic IP to trn1_cs152
<p align="center">
  <img width="70%" src="./img/aws_setup/5_name_ip.png">
</p>

5. Make sure the trn1_cs152 Elastic IP is selected (blue checkbox on the left of the name), and click the "Actions" dropdown menu in the top right of the screen.

6. Select "Associate Elastic IP address." Click on the "Instance" selection, and select the trn1_cs152 instance. Click the orange "Associate" button. 
<p align="center">
  <img width="70%" src="./img/aws_setup/6_select_instance_associate.png">
</p>

## Step 3: Setup SSH
This step is to ensure you have SSH access from your local computer to the Trn1 EC2 instance.

1. After clicking "Associate" from the previous step, you should have landed back in the Elastic IP dashboard. Scroll up on the left side of the screen until you get to "Instances" settings, and click on Instances.
2. Click on the checkbox next to your trn1_cs152 instance. Scroll down to the "Details" section and copy the Public IPv4 DNS. It should look like this: `ec2-###-###-###-###.us-east-2.compute.amazonaws.com` (where the # symbols are numbers).

<p align="center">
  <img width="70%" src="./img/aws_setup/7_get_dns.png">
</p>


3. Add the following SSH configuration to your local computer `.ssh/config`. Replace the HostName with the DNS you just copied.
```
Host trn1_cs152
    HostName ec2-###-###-###-###.us-east-2.compute.amazonaws.com
    User ubuntu
    IdentityFile ~/.ssh/trn1_cs152.pem
```
4. Now confirm you are able to ssh into the Trn1 instance. From your terminal, run:
```bash
ssh trn1_cs152
```

## Step 4: Setup Auto-Shutdown and Budget
> [!IMPORTANT] 
> 
> This final step is very important! We will set up alarms, auto-shutdown, and budget notifications to make sure you don't burn through your credits accidentally and get your credit card charged. 
>
> Note that even with these safeguards, **it is still your responsibility to make sure your instance is shutdown** in case the auto-shutdown fails. We can help resolve credit overages, payment, and billing issues before the monthly billing cycle, but once a payment is charged to your card (at then end of the month), we have no way of retroactively refunding the charge.

1. Go back to your instances dashboard on AWS console. Click on the "+" sign next to View Alarms for your trn1_cs152 instance.
2. Configure your alarm as shown in the picture below, and click Create
    - Toggle "Alarm Action" on, and Select "Stop"
    - Group samples by: Maximum
    - Type of data to sample: CPU Utilization
    - Alarm when: <=
    - Percent: 1
    - Consecutive periods: 3
    - Period: 5 minutes

<p align="center">
  <img width="70%" src="./img/aws_setup/8_alarm_details.png">
</p>

3. Search "Budgets" in the top search bar, and in the "Features" section, click on "Budgets". Click on the orange "Create a Budget" button. 
4. Fill in the information to match the picture below, and enter your email for the "Email recipients" section.
    - Select the "Monthly Cost Budget" template
    - Set "Enter your budgeted amount ($)" to 50
    - Enter your email in the "Email recipients" box

<p align="center">
  <img width="70%" src="./img/aws_setup/9_budget_details.png">
</p>

5. Finally, click the orange "Create Budget" button.

## Step 5: Apply AWS Credits
Now, we will apply the AWS credits (shoutout to AWS for their generosity!). You should have recieved in an email with the subject "[CS152] SP25 Lab 6 AWS Credit Code."

1. First, copy the AWS Promotion Code. It should be a 15 symbols long with numbers and letters, starting with "PC"

2. Go to ["Billing and Cost Management"](https://us-east-1.console.aws.amazon.com/costmanagement/home?region=us-east-2#/home). Click on "Credits" on the left side of the screen.

3. Click the orange "Redeem credit" button in the top-right of the screen.

4. Enter the Promotion Code you copied earlier, and hit "Redeem credit" 

5. Confirm the credit was applied. You should see in the "Credits" section a new entry, with the name "Berkeley CS152 Professor Christopher Fletcher", "Amount used" $0.00, and "Amount remaining" $50.00.

<p align="center">
  <img width="70%" src="./img/aws_setup/10_credit_info.png">
</p>


## Step 6: Shutdown your instance
Congrats! You have finished the setup steps for launching an AWS Tranium EC2 instance. Go back to your Instances, click on your trn1_cs152 instance, click the "Instance State" dropdown, and click "Stop instance."

> [!IMPORTANT] 
> 
> Every time you want to connect to your instance, you will need to start the instance, and once you are done working on the instance, you **must shut down your instance** to prevent extra charges.




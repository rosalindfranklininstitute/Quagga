# Quagga
Curtaining artefact quantification for electron microscopic image 

## Usage
### Installation
1. Create a new Conda environment and enter the environment   
  ```
  conda create -n quagga python=3.8.13
  conda activate quagga
  ```
  
2. Install Quagga
  ```
  pip install -e .
  ```
  
### Using Quagga
1. To run Quagga, simply use the command
  ```
  quagga.run
  ```
  
### Caveats
1. Quagga reads in a CSV as input metadata; and the CSV file _must_ have the following columns
    - "Filename" -- Path to the raw image
    - "Image name" -- Name of image
    - "Gas" -- Plasma species used in FIB milling 
    - "FIB Current (nA)" -- FIB current in nano-amperes
    - "PW (nm)" -- Pixel width of image (in nanometres)
    - "Nramps" -- Number of ramps (lamellae) in the image

# Foobar

FlexGAN is a Python library for automated synthetic relational data generation that models user provided sample data.
In three steps, you can generate synthetic data:
Step 1: Import the sample data that you wish to model, either as a Pandas dataframe or csv file.
Step 2: Inititalize training of the synthetic data generator model.
Step 3: Generate data.

## Requirements

tensorflow 2.x
pandas 1.1
numpy 1.19

## Usage

```python
import flexgan as flex

my_generator = flex.generator(csv_path='my_data.csv') # initialize flexgan by providing sampel data.
my_generator.train() # Train synthetic data generation model.
my_generator.generate_data(to_csv='my_synthetic_data.csv') # Generate synthetic data.
my_generator.save_model('my_flexgan_model.h5')
```
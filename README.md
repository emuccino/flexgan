## Description

FlexGAN is a Python library for automated synthetic relational data generation that models user provided sample data.
In three steps, you can generate synthetic data:
Step 1: Import the sample data that you wish to model, either as a Pandas dataframe or csv file.
Step 2: Initialize training of the synthetic data generator model.
Step 3: Generate data.

## Requirements

tensorflow 2.X
pandas
numpy
scikit-learn
scipy

## Usage

```python
import flexgan as flex

my_generator = flex.generator(csv_path='my_data.csv') # Initialize flexgan by providing sample data either as a pandas.DataFrame or a csv file path location.
my_generator.train() # Train synthetic data generation model.
my_generator.generate_data(to_csv='my_synthetic_data.csv') # Generate synthetic data by optionally specifying sample count and csv file path locaiton.
my_generator.save_model('my_flexgan_model.h5') # Specify path location to save a trained data generation model for future use.
my_generator = flex.generator(csv_path='my_data.csv', model_path='my_flexgan_model.h5') # Import a pretrained model to generate data.
```

## Questions or Suggestions

Please reach out if you have any questions, suggestions, or would like to contribute to the project.

email: emuccino@mindboard.com
LinkedIn: https://www.linkedin.com/in/emuccino/
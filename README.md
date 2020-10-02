![FlexGAN](https://flexlake.io/img/FlexGAN_LOGO_transparent_crop.png)

[flexlake.io](https://flexlake.io/)

## Description

FlexGAN is a Python library for automated synthetic relational data generation that models user provided sample data.
In three steps, you can generate synthetic data:

Step 1: Import the sample data that you wish to model, either as a Pandas DataFrame or csv file.

Step 2: Initialize training of the synthetic data generation model.

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

# Initialize flexgan by providing sample data either as a pandas.DataFrame or a csv file path location.
my_generator = flex.generator(csv_path='my_data.csv')

# Train synthetic data generation model.
my_generator.train()

# Generate synthetic data by optionally specifying sample count and csv file path locaiton.
my_generator.generate_data(to_csv='my_synthetic_data.csv')

# Specify path location to save a trained data generation model for future use.
my_generator.save_model('my_flexgan_model.h5')

# Import a pretrained model to generate data.
my_generator = flex.generator(csv_path='my_data.csv', model_path='my_flexgan_model.h5')
```

Check out [this](https://colab.research.google.com/github/emuccino/flexgan/blob/master/flexgan_demo.ipynb) colab notebook for an example.

## Questions or Suggestions

Please reach out if you have any questions, suggestions, or would like to contribute to the project.

email: emuccino@mindboard.com

LinkedIn: [www.linkedin.com/in/emuccino/](https://www.linkedin.com/in/emuccino/)

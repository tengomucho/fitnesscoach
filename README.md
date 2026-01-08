# Fitness Coach

This is a project that shows how it is possible to get a simple fitness coach assistant.

## Installation

To build and install the project, make sure you have Python 3.10 or newer. Then run:

```bash
pip install .
```

Or, to install in editable/development mode:

```bash
pip install -e .
```

All dependencies will be installed automatically.

## Usage

After installation, you can use the CLI tool. To get a daily fitness summary, run:

```bash
fitnesscoach summary
```

This command fetches your latest health and fitness data and outputs a concise summary.

You can get help and see all commands with:

```bash
fitnesscoach --help
```

## Login

This app uses data that is fetched from Garmin Connect, using the excellent `garminconnect` package. In order to be able to get the data, you need to login once, by doing

```bash
fitnesscoach login
```

# Run tests

To run tests you can just run

```bash
pytest -sv tests
```

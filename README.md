# Project Title

This project is a PyTorch implementation of the ACL 2018 paper: Generating Fine-Grained Open Vocabulary Entity Type Descriptions

## Prerequisites

```
Python 2.7+
PyTorch
```
##  Run

```
python <model_name>.py
```
This will complete the training, validation, and testing of the model. The results for the test data will be written to a file named "result_<model_name>.csv".
For example, to run our model, use

```
python model.py
```

## Evaluation

To run evaluation script, use

```
python evaluate.py result_<model_name>.csv
```

## Authors

* **Rajarshi Bhowmik**  - [website](https://kingsaint.github.io)
* **Gerard de Melo** - [website](https://gerard.demelo.org)

## Citation

If you use this code, please cite our paper.

```
@inproceedings{Bhowmik2018EntityDescriptions,
  title = {Generating Fine-Grained Open Vocabulary Entity Type Descriptions},
  author = {Bhowmik, Rajarshi and {de Melo}, Gerard},
  booktitle = {Proceedings of ACL 2018},
  year = {2018},
  location = {Melbourne},
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

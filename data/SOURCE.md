# Data Source

`diabetes.csv` is the **Pima Indians Diabetes Dataset**, originally compiled by the National
Institute of Diabetes and Digestive and Kidney Diseases and widely redistributed for research
and educational use via:

- UCI Machine Learning Repository: https://archive.ics.uci.edu/
- Kaggle: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

768 rows, 8 clinical features, 1 binary outcome label (`Outcome`: 1 = diabetes diagnosed
within 5 years, 0 = not diagnosed). See `reports/paper.md`, Section 3, for full column
definitions and Section 4 for how missing values (encoded as implausible zeros in several
columns) are handled.

Original citation:

Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using
the ADAP learning algorithm to forecast the onset of diabetes mellitus. In *Proceedings of the
Annual Symposium on Computer Application in Medical Care* (pp. 261–265). American Medical
Informatics Association.

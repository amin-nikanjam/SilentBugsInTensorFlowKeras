# ICSE_2022_Silent_Bugs
This Replication Package is intended for replication of results presented in the paper "Silent Bugs in Deep Learning Frameworks: An Empirical Study of Keras and Tensorflow" submitted to the 44th International Conference on Software Engineering (ICSE 2022). It contains the data and artifacts used in the paper.

The file `Keras Bugs Sheets.xlsx` contains three sheets:
* **Issues Voting**: Contains the 1168 gathered issues. For each of them, we provide title, URL, GitHub labels as well as whether it was accepted by each of the three reviewers along with comments. The total of issues for each of the number of assigned votes is given at the end of the sheet.
* **2 votes bugs**: Contains the list of the 38 two-votes bugs that were discussed by reviewers for eventual acceptation. We reported the title, URL and labels of those issues, as well as the comments of each reviewer for the issue. We highlighted a comment from the reviewer that refused it. The decision after discussion is presented in the "Accepted?" column.
* **Comparison Category**: Contains the list of the 83 issues accepted after the voting round. The same information as before is presented. We add the proposition by each reviewer for impact on the user's program (column "Scenario") as well as which part of the API was affected (column "Component"). Final decisions after group discussions are presented in "Final Scenario" and "Final Component", as well as the column "Impact" which determine the threat level according to the scale presented in our paper for each of the issues. The last column gives information about the URL of the gist for reproduction, the version(s) where the bug happened, the version that fixed the bug and eventually the commit fixing the issue.

The directory `bugs_list` contains the list of the 77 issues (.JSON format) we extracted from the GitHub API. Note that ID of issue doesn't go from 1 to 77 because we discarded 6 issues after selection (see paper for more information).

The file `data.csv` contains the list of the 77 issues with the following information: <br />
Issue number; URL; gis; Final Scenario; Final Component 1; Final Component 2; Impact; Buggy Version; Fixed Version; Fixing commit <br />
Since it is formatted as ".csv", it can be extracted easily by external tools.

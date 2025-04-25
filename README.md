# 📊 Sparx Technical Task – Data Scientist [Insights]

**Author:** Ricky Hui  
**Date:** April 2025

---

## 🚀 Overview

This repository contains my submission for the Sparx Learning technical task.

---

## 🧠 Summary of Findings

## Mandatory 

### 1.There are 1394 students data for both assessment and activity.

### 2.The mean progess is -1.9017980636237897 score, which indicated that students are not improving over time, and their marks are slightly declining.

### 3. While the distribution is approximately bell-shaped and centered near zero, it is slightly skewed to the left, reflecting a small average decline in scores. The presence of long tails on both sides suggests that some students experienced large improvements or declines, indicating potential outliers. These patterns may reflect varying levels of engagement, assessment difficulty, or learning support across the cohort.

### 4. To understand whether students really perform worse in the second test, a one-tailed paired test will be used here to find the result.

The result shows a p-value with 3.60e-41, which falls in the 0.05 interval, so we can reject H0. And there is a strong statistical evidence that students performed worse in assessment 2 compared to assessment 1.

###  5.Using bootstrapping, we find that the average score decreased between assessments.Since the entire confidence interval is negative, we are highly confident that this is not due to random variation.



### Exploratory

To explore whether using Sparx improves students' exam performance, we analyzed engagement with support videos — a key self-learning feature of the platform. Students were grouped into high and low engagement categories based on video time watched. Using a Welch’s t-test, we found that high-engagement students performed significantly better in assessments (p = 0.0069).

This suggests a positive association between Sparx usage and academic outcomes, supporting the claim that increased engagement with Sparx resources contributes to improved performance. However, as this is observational data, causality cannot be confirmed without further controlled studies.


### Stretch
High-performing students often demonstrate strong initiative and can effectively utilize the learning platform with minimal intervention. Therefore, I suggest that a personalized model should focus more on student engagement levels. Higher engagement typically reflects a greater willingness to learn, increased motivation, and a higher likelihood of sustained success. By tailoring homework based on engagement signals—such as time spent on support videos or repeated attempts at challenging tasks—we can provide the right level of support to encourage consistent progress and long-term academic growth.


To achieve this, I propose building a probabilistic machine learning model, inspired by Bayesian Knowledge Tracing, that estimates each student’s mastery of various topics using engagement data alone. The model would use features such as:

- Time spent per task or question

- Number of video views

- Frequency of correct vs. incorrect attempts

- Time to first correct answer

- Question repetition and revisit patterns

The model would output a predicted mastery level for each topic, which could then guide homework personalization by:

- Assigning reinforcement tasks for students predicted to struggle

- Introducing more advanced problems for students with high predicted mastery

- Balancing revision with new learning for mid-level students

Over time, the model can be retrained as more data becomes available, making it increasingly accurate. By integrating this with a lightweight deployment (via API or container), it could be scaled efficiently across the platform and serve real-time recommendations with minimal overhead.

---

## 📁 Project Structure

```
sparx-technical-task/
├── README.md
├── requirements.txt
├── analysis.ipynb
├── helpers.py
├── data/
│   └── sparx_data.csv (if provided)
```

---

## 💻 Setup Instructions

---

### 1. Clone this repo:
```bash
git clone https://github.com/rickyhui28/sparx-technical-task.git
cd sparx-technical-task
```

### 2. Create a virtual environment:
```bash
python3 -m venv env
source env/bin/activate  # Windows: .\env\Scripts\activate
```

### 3. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook:
```bash
make notebook
```

### 5. (Optional) Export HTML summary:
```bash
make export-summary
```

---


## 📬 Contact

If you have any issues running the code or questions, feel free to contact me.

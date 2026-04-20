Resume Matcher (NLP-Based)

## Overview
This project implements a Resume Matching System using Natural Language Processing (NLP) techniques to rank resumes based on user-defined job requirements. The system combines semantic similarity with explicit skill matching to produce accurate and interpretable results.

## Features
* Semantic matching using TF-IDF with n-grams
* Skill extraction using rule-based NLP
* Hybrid scoring mechanism combining semantic similarity and skill overlap
* Resume filtering based on experience, degree, and field of study
* Interactive interface built with Streamlit

## Technology Stack
* Programming Language: Python
* Libraries: Pandas, Scikit-learn
* Interface: Streamlit
* Core Concepts: TF-IDF, Cosine Similarity, Text Preprocessing

## Dataset
This project uses a resume dataset sourced from Kaggle.
Dataset link:
https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset

## Installation and Setup
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install pandas scikit-learn streamlit

Run the application
streamlit run frontend/app.py

## Author
Prakhar Pant

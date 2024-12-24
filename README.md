# Enhancing-Zero-Shot-Crypto-Sentiment-With-Fine-Tuned-Language-Model-and-Prompt-Engineering

## Abstract

Blockchain technology has revolutionized the financial landscape, witnessing widespread adoption of cryptocurrencies due to their decentralized and transparent nature. As sentiments expressed on social media platforms wield substantial influence over cryptocurrency market dynamics, sentiment analysis has emerged as a crucial tool for gauging public opinion and predicting market trends. This paper explores fine-tuning techniques for large language models to enhance sentiment analysis performance. Experimental results demonstrate a significant average zero-shot performance gain of 40% on unseen tasks after fine-tuning, highlighting its potential. Additionally, the impact of instruction-based fine-tuning on models of varying scales is examined, revealing that larger models benefit from instruction tuning, achieving the highest average accuracy score of 75.16%. In contrast, smaller-scale models may experience reduced generalization due to complete model capacity utilization. To gain deeper insight into instruction effectiveness, the paper presents experimental investigations under different instruction tuning setups. Results show the model achieves an average accuracy score of 72.38% for short and simple instructions, outperforming long and complex instructions by over 12%. Finally, the paper explores the relationship between fine-tuning corpus size and model performance, identifying an optimal corpus size of 6,000 data points for the highest performance across different models. Microsoft’s MiniLM, a distilled version of BERT, excels in efficient data use and performance optimization, while Google’s FLAN-T5 demonstrates consistent and reliable performance across diverse datasets.

## Overview of The Proposed System Architecture

<img src="images/Propose_diagram_color.png" width="1280px" height="720px" />


Robustness Evaluation Metrics

Quick Start

Installation

Dataset

Experiment

Data Analysis

Acknowledgment

## Citation

If you use this work in your research or find it helpful, please cite:

```bibtex
@article{wahidur2024enhancing,
  title={Enhancing zero-shot crypto sentiment with fine-tuned language model and prompt engineering},
  author={Wahidur, Rahman SM and Tashdeed, Ishmam and Kaur, Manjit and Lee, Heung-No},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}

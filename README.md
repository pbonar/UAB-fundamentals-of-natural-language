# Negation and Uncertainty Detection in Clinical Texts

## Description

This project focuses on detecting negation and uncertainty expressions in clinical narratives written in Spanish and Catalan. It includes both a rule-based system and a machine learning approach to improve detection accuracy. For detailed insights, please refer to the report located at:

`Project/raport/Negation_and_Uncertainty_Detection_using_Classical_and_Machine_Learning_Techniques.pdf`


## Project Structure

- **Problems**  
  - `L2_Basic_Text_Processing.pdf`
  - `L3_Syntactic_Parsing.pdf`
  - `L4_Language_Model_Problems.pdf`
  - `L5_Sequence_Labeling.pdf`

- **Project**  
  - **code**  
    - `evaluator.py`
    - `negation-detector_machine-learning.ipynb`
    - `negation-detector_rule-based.py`
  - **presentation**  
    - `Negation and Uncertainty Detection using Classical and Machine.pptx`
    - `script.pdf`
  - **raport**  
    - `Negation_and_Uncertainty_Detection_using_Classical_and_Machine_Learning_Techniques.pdf`
  - **resources**  
    - `negacio_test_v2024.json`
    - `negacio_train_v2024.json`

## Authors

- Piotr Bonar  
- Iker Romero Cespedesb  
- Miriam Morales Francoc  
- Suzana Jeal  
- Adnan Boukfal Lazaare

## Usage

### Running the Rule-Based Detector

```bash
python3 Project/code/negation-detector_rule-based.py --input Project/resources/negacio_test_v2024.json





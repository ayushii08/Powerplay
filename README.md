# Powerplay AI 
The solution focuses on building a reliable LLM-based extraction system for construction procurement inputs, with an emphasis on structured outputs, minimal hallucinations, and safe handling of ambiguous or incomplete data.

## Repository Contents

- `solution.py` — Main implementation for Task 2  
- `test_inputs.txt` — Custom test dataset covering edge cases  
- `outputs.json` — Generated structured outputs from the model  
- `design_explanation.pdf` — Task 1: LLM workflow design explanation  
- `evaluation_notes.pdf` — Task 3 & 4: Edge case evaluation and reflection  

## How to Run

### Requirements
- Python 3.9+
- OpenAI API key

### Setup
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_api_key
```
## Run the Extractor

```bash
python solution.py
```

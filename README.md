# 💬 Gemma-2B Customer Support Chatbot

*A fine-tuned language model that aims to make customer-support replies more helpful, consistent, and safer.*

Have you ever needed a quick answer about a refund, a cancelled order, or a damaged product? This project explores how a small language model can be adapted to answer common customer-support questions.

I fine-tuned **Google Gemma-2B** on the **Bitext Customer Support dataset** using **QLoRA**, then added lightweight output guardrails and evaluated generated responses with ROUGE metrics.

> **Project format:** Google Colab notebook  
> **Main notebook:** `Customer_Support_Chatbot.ipynb`

## ✨ What this project does

- Learns from customer-support instruction and response examples.
- Uses 4-bit quantization to reduce the memory needed to fine-tune the base model.
- Trains LoRA adapter weights rather than updating every parameter in Gemma-2B.
- Generates replies to common questions about orders, refunds, cancellations, and damaged products.
- Applies simple output guardrails, including handling empty responses and limiting response length.
- Calculates ROUGE scores against reference responses and exports an evaluation report as a CSV file.

## 🧰 Tech stack

| Tool | How it is used |
|---|---|
| Python | Main programming language |
| Google Colab | Notebook-based development and GPU training |
| PyTorch | Model execution and tensor operations |
| Hugging Face Transformers | Loading Gemma-2B and its tokenizer |
| Hugging Face Datasets | Loading the Bitext dataset |
| PEFT | Preparing and training LoRA adapters |
| BitsAndBytes | 4-bit model quantization |
| TRL (`SFTTrainer`) | Supervised fine-tuning |
| Evaluate / ROUGE | Comparing generated responses with reference answers |
| Pandas | Creating the evaluation CSV report |

## 🧠 How it works

1. **Load the data** — fetch the Bitext Customer Support dataset from Hugging Face.
2. **Prepare the examples** — split the data into training and validation sets, then format each example as an instruction/response pair.
3. **Load Gemma-2B** — load the pretrained model and tokenizer. Access to the model may require accepting its Hugging Face terms.
4. **Reduce memory use** — load the model in 4-bit NF4 quantization with double quantization enabled.
5. **Configure QLoRA** — prepare the quantized model for training and attach LoRA adapters to the attention projection layers.
6. **Fine-tune** — train with the supervised fine-tuning trainer.
7. **Save and reload** — save the adapter and tokenizer, then reload them for inference.
8. **Generate answers** — test the model with example customer-support questions.
9. **Apply guardrails** — run the generated answer through simple output checks.
10. **Evaluate** — calculate ROUGE metrics and export the results to `evaluation_report.csv`.

## 📁 Repository structure

Keep the notebook and README in the repository root:

```text
customer-support-chatbot/
├── Customer_Support_Chatbot.ipynb
├── README.md
└── evaluation_report.csv       # optional: add after generating it
```

The notebook saves the trained adapter and evaluation report to Google Drive. Those outputs do not automatically appear in this GitHub repository; download or copy them into the repository if you want to publish them.

## 🚀 Run the notebook

### 1. Open it in Google Colab

Upload `Customer_Support_Chatbot.ipynb` to Google Colab, or open it from your GitHub repository once it has been pushed.

### 2. Choose a GPU runtime

In Colab, select **Runtime → Change runtime type → T4 GPU** (or another compatible GPU if available). Free GPU availability and memory limits can vary.

### 3. Configure Hugging Face access

The notebook reads `HF_TOKEN` from Colab Secrets. Add your Hugging Face access token there if required, and enable notebook access to that secret. Do **not** paste your token into the notebook or commit it to GitHub.

You may also need to visit the Gemma model page on Hugging Face, sign in, and accept the model's access terms before downloading it.

### 4. Run the cells in order

Run the notebook from top to bottom. It installs libraries, loads the dataset and model, trains the adapter, saves model files, tests responses, applies guardrails, and produces evaluation results.

**Note:** Training can take time and consume substantial GPU memory. The notebook currently selects 3,000 training examples and 300 validation examples to keep the experiment smaller. Changing the sample size, package versions, or GPU can change results and compatibility.

## 🧪 Evaluation

The notebook uses **ROUGE** metrics to compare generated answers with reference answers and saves the resulting scores to `evaluation_report.csv`.

ROUGE is a useful text-overlap measure, but it does not prove that an answer is factually correct, safe, or genuinely helpful. The results should be interpreted alongside manual review and realistic test cases. Add your actual evaluation scores here after running the notebook—avoid reporting scores that you have not measured.

## 🛡️ Guardrails and limitations

The notebook includes a lightweight `apply_guardrails()` function and a `safe_generate_response()` wrapper. These are basic output checks, not a complete safety system.

- The model may still produce incorrect, incomplete, or irrelevant answers.
- ROUGE scores measure overlap with references, not customer satisfaction or factual correctness.
- The project is an experimental portfolio project, not a production-ready customer-support service.
- Do not use it to make real refund, payment, account, or order decisions without verified business data and appropriate safeguards.
- Never upload Hugging Face tokens, private customer data, or other secrets to a public repository.

## 🔭 Possible next steps

- Add a small, diverse test set and inspect responses manually.
- Compare the fine-tuned model with the original Gemma-2B baseline.
- Track latency, memory usage, and quality alongside ROUGE.
- Add stronger checks for unsupported claims, sensitive information, and off-topic responses.
- Build a simple Streamlit interface for interactive testing.
- Document the actual results and include a few representative examples.

## 👋 Feedback

Thanks for checking out this project! If you have suggestions for improving the evaluation, guardrails, or user experience, feel free to open an issue or share feedback.

---

**Made as a learning and portfolio project in fine-tuning, parameter-efficient training, and LLM evaluation.**

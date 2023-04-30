from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,AutoModelForMaskedLM
import torch
import gradio as gr



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
new_model = AutoModelForSeq2SeqLM.from_pretrained("./Exp_large_code_t5-base")
# new_model = AutoModelForMaskedLM.from_pretrained("neulab/codebert-python")
new_tokenizer = AutoTokenizer.from_pretrained("./Exp_large_code_t5-base", use_fast=True)


# new_tokenizer = AutoTokenizer.from_pretrained("neulab/codebert-python")
#
# new_model = AutoModelForMaskedLM.from_pretrained("neulab/codebert-python")
#
def generate_code(NL):
    inputs = new_tokenizer([NL], padding="max_length", truncation=True, max_length=64, return_tensors="pt")

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    outputs = new_model.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_code = new_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(output_code)
    return output_code[0]
# def generate_code(NL):
#     n = NL + " " + "<mask>"
#     inputs = new_tokenizer(n,  return_tensors="pt")
#
#     input_ids = inputs.input_ids
#     attention_mask = inputs.attention_mask
#     with torch.no_grad():
#         outputs = new_model(**inputs).logits
#
#     mask_token_index = (inputs.input_ids == new_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
#
#     predicted_token_id = outputs[0, mask_token_index].argmax(axis=-1)
#     print(predicted_token_id)
#     output_code = new_tokenizer.decode(predicted_token_id)
#     # all special tokens including will be removed
#     # output_code = new_tokenizer.decode(outputs, skip_special_tokens=True)
#     print(output_code)
#     return output_code

output_text = gr.outputs.Textbox()
gr.Interface(generate_code,"textbox", output_text, title="Python Code Generation", description="Enter the NL Text Input", allow_flagging='never').launch(server_name="0.0.0.0", server_port=7860)
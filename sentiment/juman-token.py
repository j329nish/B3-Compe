import subprocess
import pandas as pd
import json

def jumanpp_tokenize(input_text):
    process = subprocess.Popen(['jumanpp', '--segment'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=input_text.encode('utf-8'))
    if process.returncode != 0:
        raise Exception(f"Jumanpp error: {stderr.decode('utf-8')}")
    return stdout.decode('utf-8')

def save_tokenized_texts(input_texts):
    tokenized_text = [jumanpp_tokenize(i) for i in input_texts]
    df = pd.DataFrame(tokenized_text, columns=["sentence"])
    return df

if __name__ == "__main__":
    input_file = "dataset/train.json"
    sentences = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            dsss = json.loads(line)
            sentences.append(dsss["sentence"])

    df = save_tokenized_texts(sentences)
    print(df)
    output_file = "dataset/train-tokenized.json"
    df.to_json(output_file, orient='records', lines=True, force_ascii=False)
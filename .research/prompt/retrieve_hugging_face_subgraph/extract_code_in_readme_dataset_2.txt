
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
tags:
- Lilac
---
# lilac/TruthfulQA-MultipleChoice
This dataset is a [Lilac](http://lilacml.com) processed dataset. Original dataset: [https://huggingface.co/datasets/truthful_qa](https://huggingface.co/datasets/truthful_qa)

To download the dataset to a local directory:

```bash
lilac download lilacai/lilac-TruthfulQA-MultipleChoice
```

or from python with:

```py
ll.download("lilacai/lilac-TruthfulQA-MultipleChoice")
```


Output:
{
    "extracted_code": "ll.download(\"lilacai/lilac-TruthfulQA-MultipleChoice\")"
}

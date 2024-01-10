
import argparse
import transformers as tr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "spm_model_file",
        type=str,
    )
    parser.add_argument(
        "output_dir",
        type=str,
    )
    args = parser.parse_args()

    # assumes a LLamaTokenizer

    tokenizer = tr.LlamaTokenizer(vocab_file=args.spm_model_file)
    tokenizer.save_pretrained(args.output_dir)

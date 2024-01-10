import argparse
import sentencepiece as spm
import transformers as tr
import time 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
    )
    args = parser.parse_args()

    start = time.time()
    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=32768,
        character_coverage=0.9999,       
        model_type="bpe",       
        byte_fallback=True,       
        split_digits=True,
        num_threads=32
    )
    end = time.time()
    print("Training took {} seconds".format(end - start))

    # assumes a llama tokenizer
    print("Saving HF tokenizer")
    hf_tokenizer = tr.LlamaTokenizer(args.model_prefix + ".model")
    hf_tokenizer.save_pretrained(args.model_prefix + "-hf")
